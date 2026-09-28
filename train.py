"""Обучение, валидация и сабмит.

    python train.py            # признаки -> валидация всех ступеней -> submission.csv
    python train.py --no-cv    # только обучение на всём train и сабмит
"""

from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import add_target_features, build_features, load, neighbor_score_features
from metric import precision_at_recall

SEEDS = [0, 1, 2, 3, 4]
LGB_PARAMS = dict(
    objective="binary", learning_rate=0.02, n_estimators=800, num_leaves=31,
    min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.3,
    reg_lambda=5.0, verbose=-1,
)
# Тест идёт сразу после трейна по времени (20–26 апреля), поэтому валидация — по времени:
# учимся на всех днях до фолда, проверяем на следующих днях.
TIME_FOLDS = [("2026-04-12", "2026-04-14"), ("2026-04-14", "2026-04-16"),
              ("2026-04-16", "2026-04-18"), ("2026-04-18", "2026-04-20")]
# Сплит «как тест»: 7 дней валидации подряд после обучающих. Проверяет, как держится
# качество, когда последняя известная разметка отстаёт на неделю.
HORIZON_SPLIT = ("2026-04-13", "2026-04-20")
BASELINE_COLS = ["n_events", "item_nunique"]        # признаки из quickstart.ipynb


# ----------------------------------------------------------------------------- модели
# Все модели имеют одну сигнатуру: (X_all, pairs, labeled, target_ids) -> score, где
# labeled — размеченные куки, которые разрешено использовать (cookie_id, target), а
# target_ids — куки, для которых нужен скор.

def _lgb(X_tr, y_tr, X_te, seeds=SEEDS):
    """LightGBM, усреднённый по нескольким сидам (снижает дисперсию ранжирования)."""
    pred = np.zeros(len(X_te))
    for s in seeds:
        model = lgb.LGBMClassifier(**LGB_PARAMS, random_state=s).fit(X_tr, y_tr)
        pred += model.predict_proba(X_te)[:, 1] / len(seeds)
    return pred


def baseline(X_all, pairs, labeled, target_ids):
    """Baseline из quickstart: RandomForest на двух счётчиках."""
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1)
    model.fit(X_all.loc[labeled.cookie_id, BASELINE_COLS], labeled.target)
    return model.predict_proba(X_all.loc[target_ids, BASELINE_COLS])[:, 1]


def behavior_only(X_all, pairs, labeled, target_ids):
    """LightGBM только на поведенческих признаках куки (без информации о других куках)."""
    cols = [c for c in X_all.columns if not c.startswith("pop_")]
    return _lgb(X_all.loc[labeled.cookie_id, cols], labeled.target.values, X_all.loc[target_ids, cols])


def one_stage(X_all, pairs, labeled, target_ids, seeds=SEEDS):
    """+ популярность объявлений и их «ботовость» по размеченной истории."""
    F = add_target_features(X_all, pairs, labeled.set_index("cookie_id").target)
    return _lgb(F.loc[labeled.cookie_id], labeled.target.values, F.loc[target_ids], seeds)


def two_stage(X_all, pairs, labeled, target_ids, seeds=SEEDS):
    """+ бот-скор соседей по объявлениям.

    1-й уровень: one_stage-модель скорит ВСЕ куки (включая неразмеченные и test). Для
    размеченных кук берём out-of-fold скор, иначе у соседей-из-train был бы скор модели,
    видевшей их метку. 2-й уровень учится на признаках + neighbor_score_features.
    """
    F = add_target_features(X_all, pairs, labeled.set_index("cookie_id").target)
    ids, y = labeled.cookie_id.values, labeled.target.values
    score = pd.Series(_lgb(F.loc[ids], y, F, seeds=SEEDS[:1]), index=F.index)
    oof = np.zeros(len(ids))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y):
        oof[b] = _lgb(F.loc[ids[a]], y[a], F.loc[ids[b]], seeds=SEEDS[:1])
    score.loc[ids] = oof
    F2 = F.join(neighbor_score_features(pairs, score))
    return _lgb(F2.loc[ids], y, F2.loc[target_ids], seeds)


# ----------------------------------------------------------------------------- валидация

def _split(model, X_all, pairs, train, a, b):
    start = train.window_start_ts
    va = train[(start >= a) & (start < b)]
    return va, model(X_all, pairs, train[start < a], va.cookie_id.values)


def time_cv(model, X_all, pairs, train):
    """Расширяющееся окно: P@R0.7 по фолдам, на объединённом OOF и на 7-дневном сплите."""
    parts = [_split(model, X_all, pairs, train, a, b) for a, b in TIME_FOLDS]
    folds = [precision_at_recall(va.target, p) for va, p in parts]
    y = np.concatenate([va.target.values for va, _ in parts])
    p = np.concatenate([p for _, p in parts])
    va, ph = _split(model, X_all, pairs, train, *HORIZON_SPLIT)
    return dict(folds=np.round(folds, 3).tolist(), mean_fold=float(np.mean(folds)),
                pooled=precision_at_recall(y, p), pr_auc=average_precision_score(y, p),
                roc_auc=roc_auc_score(y, p), horizon=precision_at_recall(va.target, ph))


# ----------------------------------------------------------------------------- main

def main(run_cv: bool = True, out_path: str = "submission.csv"):
    t0 = time.time()
    train, test, events = load()
    meta = pd.concat([train, test], ignore_index=True)
    # поведенческие признаки куки считаются только по её собственным событиям в её окне;
    # популярность объявлений — по событиям других кук не позже конца окна (без меток)
    X_all, pairs = build_features(events, meta)
    print(f"признаки без меток: {X_all.shape[1]}, train {len(train)}, test {len(test)}  "
          f"({time.time() - t0:.0f} c)", flush=True)

    if run_cv:
        fast = dict(seeds=SEEDS[:1])
        for name, model in [
            ("baseline RF (2 признака)", baseline),
            ("LightGBM: поведение", behavior_only),
            ("LightGBM: + объявления", lambda *a: one_stage(*a, **fast)),
            ("LightGBM: + объявления + соседи (финал)", lambda *a: two_stage(*a, **fast)),
        ]:
            r = time_cv(model, X_all, pairs, train)
            print(f"\n{name}\n  P@R0.7 по фолдам {r['folds']}, среднее {r['mean_fold']:.4f}; "
                  f"объединённый OOF {r['pooled']:.4f} (PR-AUC {r['pr_auc']:.4f}, ROC-AUC {r['roc_auc']:.4f}); "
                  f"7-дневный сплит {r['horizon']:.4f}  ({time.time() - t0:.0f} c)", flush=True)

    # финал: вся разметка train (для test это строго прошлые дни)
    score = two_stage(X_all, pairs, train[["cookie_id", "target"]], test.cookie_id.values)
    sub = pd.DataFrame({"cookie_id": test.cookie_id.values, "score": np.clip(score, 0, 1)})
    assert len(sub) == len(test) and sub.cookie_id.is_unique and sub.score.notna().all()
    sub.to_csv(out_path, index=False)
    print(f"\n{out_path} сохранён: {len(sub)} строк  ({time.time() - t0:.0f} c)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cv", action="store_true", help="пропустить валидацию")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()
    main(run_cv=not args.no_cv, out_path=args.out)
