"""Сборка признаков куки из потока событий внутри суточного окна наблюдения."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

SESSION_GAP_S = 30 * 60          # пауза, после которой считаем, что началась новая сессия
EVENT_NAMES = [
    "search_results_view", "item_view", "photo_swipe", "seller_page_view",
    "contact_phone_show", "contact_chat_open", "contact_message_sent",
    "favorite_add", "login",
]   # captcha_shown не берём: внутри окна его нет ни у одной куки, все показы — после окна
SCRIPT_UA = re.compile(r"curl|python|go-http|scrapy|node-fetch|wget|java/", re.I)
PLATFORM_MAP = {"web": "web", "desktop": "web", "android": "android", "ios": "ios", "iphone": "ios"}


# ----------------------------------------------------------------------------- загрузка

def load(data_dir: str = "data"):
    dates = ["cookie_created_at", "window_start_ts", "window_end_ts"]
    train = pd.read_csv(f"{data_dir}/train.csv", parse_dates=dates)
    test = pd.read_csv(f"{data_dir}/test.csv", parse_dates=dates)
    events = pd.read_csv(f"{data_dir}/events.csv.gz", parse_dates=["event_ts"])
    return train, test, events


def clean_events(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Дедупликация, нормализация, отсечение по окну и сортировка по времени.

    * Полные дубликаты строк — артефакт логирования (доля одинакова у ботов и людей),
      удаляем, иначе они искажают счётчики и дают нулевые интервалы.
    * `platform` записан в разном регистре/синонимах (WEB/Web/desktop, iOS/iphone) —
      приводим к трём значениям.
    * События вне [window_start_ts, window_end_ts) выбрасываем: в файле есть события
      после окна, и у ботов их заметно больше — использовать их означало бы утечку.
    * Файл не отсортирован — сортируем внутри куки по времени (stable, чтобы при равном
      времени сохранить исходный порядок).
    """
    ev = events.drop_duplicates().copy()
    ev["platform"] = ev["platform"].str.lower().map(PLATFORM_MAP).fillna("other")
    ev = ev.merge(meta[["cookie_id", "window_start_ts", "window_end_ts"]], on="cookie_id")
    ev = ev[(ev.event_ts >= ev.window_start_ts) & (ev.event_ts < ev.window_end_ts)]
    ev = ev.sort_values(["cookie_id", "event_ts"], kind="mergesort").reset_index(drop=True)
    return ev


# ----------------------------------------------------------------------------- признаки

def _safe_div(a, b):
    return np.where(np.asarray(b) > 0, np.asarray(a, dtype=float) / np.maximum(b, 1e-9), 0.0)


def timing_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Признаки по моментам времени: ритм, регулярность, сессии, время суток."""
    ev = ev[["cookie_id", "event_ts", "window_start_ts"]].copy()
    ev["gap"] = ev.groupby("cookie_id").event_ts.diff().dt.total_seconds()
    ev["log_gap"] = np.log1p(ev["gap"])
    ev["new_session"] = (ev["gap"].isna() | (ev["gap"] > SESSION_GAP_S)).astype(int)
    ev["session"] = ev.groupby("cookie_id").new_session.cumsum()
    hour = ev.event_ts.dt.hour + ev.event_ts.dt.minute / 60
    ev["night"] = (hour < 6).astype(int)
    ev["hour"] = hour
    ev["sec"] = ev.event_ts.dt.second

    g = ev.groupby("cookie_id")
    gaps = ev.dropna(subset=["gap"])
    in_sess = gaps[gaps.gap <= SESSION_GAP_S]          # интервалы внутри сессий: темп работы
    gg, gs = gaps.groupby("cookie_id"), in_sess.groupby("cookie_id")

    f = pd.DataFrame(index=g.size().index)
    first, last = g.event_ts.min(), g.event_ts.max()
    f["span_min"] = (last - first).dt.total_seconds() / 60
    f["first_offset_h"] = (first - g.window_start_ts.first()).dt.total_seconds() / 3600
    f["n_sessions"] = g.session.max()
    sess = ev.groupby(["cookie_id", "session"]).event_ts.agg(["size", "min", "max"])
    sess["dur"] = (sess["max"] - sess["min"]).dt.total_seconds()
    sg = sess.groupby(level=0)
    f["max_session_events"] = sg["size"].max()
    f["max_session_dur_min"] = sg["dur"].max() / 60
    f["active_min"] = sg["dur"].sum() / 60
    # «самая скриптовая» сессия: у куки может быть и обычная активность, и автоматическая,
    # поэтому помимо общих агрегатов берём экстремум по сессиям (сессии от 3 интервалов)
    in_sess_ev = ev[ev.gap.notna() & (ev.gap <= SESSION_GAP_S)]
    ps = in_sess_ev.groupby(["cookie_id", "session"]).gap.agg(
        n="size", med="median", fast=lambda s: s.between(3, 20, inclusive="left").mean())
    ps = ps[ps.n >= 3].groupby(level=0)
    f["sess_min_gap_median"] = ps.med.min()
    f["sess_max_fast_share"] = ps.fast.max()

    f["gap_min"] = gg.gap.min()
    f["gap_median"] = gg.gap.median()
    f["gap_max"] = gg.gap.max()
    f["frac_gap0"] = gg.gap.apply(lambda s: (s == 0).mean())
    f["frac_gap_le3"] = gg.gap.apply(lambda s: (s <= 3).mean())
    f["frac_gap_long"] = gg.gap.apply(lambda s: (s > SESSION_GAP_S).mean())
    f["s_gap_mean"] = gs.gap.mean()
    f["s_gap_median"] = gs.gap.median()
    f["s_gap_std"] = gs.gap.std()
    f["s_gap_cv"] = f["s_gap_std"] / f["s_gap_mean"].replace(0, np.nan)
    f["s_loggap_std"] = gs.log_gap.std()
    f["s_gap_q10"] = gs.gap.quantile(0.1)
    f["s_gap_q90"] = gs.gap.quantile(0.9)
    # у скриптов интервалы сидят в узкой полосе 3–20 с, у людей размазаны по 15–120 с
    f["s_gap_fast_share"] = gs.gap.agg(lambda s: s.between(3, 20, inclusive="left").mean())
    f["s_gap_slow_share"] = gs.gap.agg(lambda s: (s >= 30).mean())
    # доля самого частого значения интервала — «метроном» скрипта с фиксированной задержкой
    f["s_gap_mode_share"] = gs.gap.agg(lambda s: s.value_counts(normalize=True).iloc[0] if len(s) else np.nan)
    f["s_gap_nunique_ratio"] = gs.gap.nunique() / gs.size()
    # автокорреляция соседних интервалов: «пачки» быстрых действий vs независимые задержки
    f["s_gap_autocorr"] = gs.log_gap.agg(lambda s: s.autocorr() if len(s) > 3 else np.nan)
    f["events_per_active_min"] = _safe_div(g.size(), f["active_min"] + 1)

    f["night_share"] = g.night.mean()
    f["hour_mean"] = g.hour.mean()
    f["hour_std"] = g.hour.std()
    f["n_hours"] = g.event_ts.agg(lambda s: s.dt.hour.nunique())
    # разнообразие секунды внутри минуты: запуск по расписанию даёт повторяющиеся секунды
    f["sec_nunique_ratio"] = g.sec.nunique() / g.size()
    return f


def content_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Что смотрит кука: типы событий, разнообразие объявлений/категорий/запросов, выдача."""
    g = ev.groupby("cookie_id")
    n = g.size()
    f = pd.DataFrame(index=n.index)
    f["n_events"] = n

    cnt = pd.crosstab(ev.cookie_id, ev.event_name).reindex(columns=EVENT_NAMES, fill_value=0)
    for c in EVENT_NAMES:
        f[f"n_{c}"] = cnt[c]
        f[f"share_{c}"] = cnt[c] / n
    f["n_event_types"] = g.eid.nunique()
    contacts = cnt[["contact_phone_show", "contact_chat_open", "contact_message_sent"]].sum(axis=1)
    iv = cnt["item_view"]
    f["contact_per_view"] = _safe_div(contacts, iv)
    f["photo_per_view"] = _safe_div(cnt["photo_swipe"], iv)
    f["seller_per_view"] = _safe_div(cnt["seller_page_view"], iv)
    f["fav_per_view"] = _safe_div(cnt["favorite_add"], iv)
    f["view_per_search"] = _safe_div(iv, cnt["search_results_view"])

    # разнообразие
    f["item_nunique"] = g.item_id.nunique()
    f["cat_nunique"] = g.item_category.nunique()
    f["loc_nunique"] = g.item_location.nunique()
    f["query_nunique"] = g.search_query.nunique()
    f["item_uniq_ratio"] = _safe_div(f["item_nunique"], g.item_id.count())
    f["items_per_cat"] = _safe_div(f["item_nunique"], f["cat_nunique"])
    views = ev[ev.event_name == "item_view"].groupby("cookie_id")
    f["view_repeat_ratio"] = (1 - views.item_id.nunique() / views.size()).reindex(f.index)
    f["cat_top_share"] = g.item_category.agg(
        lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else np.nan)
    f["seller_pro_share"] = g.seller_type.agg(
        lambda s: (s == "pro").mean() if s.notna().any() else np.nan)
    f["share_no_category"] = g.item_category.apply(lambda s: s.isna().mean())

    # выдача: глубина листания и «пролистывание подряд» (стр. k -> k+1 по тому же запросу)
    sr = ev[ev.event_name == "search_results_view"].copy()
    sr["prev_page"] = sr.groupby("cookie_id").search_page.shift()
    sr["prev_q"] = sr.groupby("cookie_id").search_query.shift()
    sr["seq_page"] = ((sr.search_page == sr.prev_page + 1) & (sr.search_query == sr.prev_q)).astype(int)
    sg = sr.groupby("cookie_id")
    f["page_max"] = sg.search_page.max().reindex(f.index)
    f["page_mean"] = sg.search_page.mean().reindex(f.index)
    f["frac_page_gt3"] = sg.search_page.apply(lambda s: (s > 3).mean()).reindex(f.index)
    f["seq_page_share"] = sg.seq_page.mean().reindex(f.index)
    f["searches_per_query"] = (sg.size() / sg.search_query.nunique().replace(0, np.nan)).reindex(f.index)
    f["query_len_mean"] = sg.search_query.agg(lambda s: s.str.len().mean()).reindex(f.index)
    return f


def sequence_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Порядок действий: время «чтения» страниц и однообразие переходов."""
    ev = ev[["cookie_id", "event_ts", "event_name", "item_location"]].copy()
    grp = ev.groupby("cookie_id")
    ev["next_gap"] = -grp.event_ts.diff(-1).dt.total_seconds()
    ev["prev_name"] = grp.event_name.shift()
    ev["same_as_prev"] = (ev.event_name == ev.prev_name).astype(float)
    prev_loc = grp.item_location.shift()
    ev["loc_switch"] = ((ev.item_location != prev_loc) & ev.item_location.notna() & prev_loc.notna()).astype(float)
    # паузы > 30 мин — это уход, а не чтение страницы
    ev.loc[ev.next_gap > SESSION_GAP_S, "next_gap"] = np.nan

    f = pd.DataFrame(index=grp.size().index)
    for name, short in [("item_view", "item"), ("search_results_view", "serp"), ("photo_swipe", "photo")]:
        d = ev[ev.event_name == name].groupby("cookie_id").next_gap
        f[f"dwell_{short}_median"] = d.median()
        f[f"dwell_{short}_q10"] = d.quantile(0.1)
    views = ev[ev.event_name == "item_view"].groupby("cookie_id").prev_name
    f["view_after_view"] = views.apply(lambda s: (s == "item_view").mean())
    f["view_after_serp"] = views.apply(lambda s: (s == "search_results_view").mean())
    g = ev.groupby("cookie_id")
    f["same_type_share"] = g.same_as_prev.mean()
    f["loc_switch_share"] = g.loc_switch.mean()
    f["loc_top_share"] = grp.item_location.agg(
        lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else np.nan)
    return f


def pointer_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Курсор (есть только на web): наличие, разброс, повторяемость координат."""
    web = ev[ev.platform == "web"]
    g = web.groupby("cookie_id")
    f = pd.DataFrame(index=ev.cookie_id.unique())
    f["web_share"] = ev.groupby("cookie_id").platform.agg(lambda s: (s == "web").mean())
    f["ptr_share"] = g.pointer_x.apply(lambda s: s.notna().mean())
    p = web.dropna(subset=["pointer_x"])
    pg = p.groupby("cookie_id")
    f["ptr_n"] = pg.size()
    f["ptr_x_mean"] = pg.pointer_x.mean()
    f["ptr_y_mean"] = pg.pointer_y.mean()
    # у людей x равномерен по всей ширине 1920, у скриптов прижат к левой части
    # (узкий headless-viewport), поэтому смотрим на верхние квантили, а не только на среднее
    f["ptr_x_q90"] = pg.pointer_x.quantile(0.9)
    f["ptr_y_q90"] = pg.pointer_y.quantile(0.9)
    f["ptr_x_right_share"] = (p.pointer_x > 1366).groupby(p.cookie_id).mean()
    f["ptr_center_dist"] = np.hypot(p.pointer_x - 960, p.pointer_y - 540).groupby(p.cookie_id).mean()
    f["ptr_x_std"] = pg.pointer_x.std()
    f["ptr_y_std"] = pg.pointer_y.std()
    f["ptr_uniq_ratio"] = (p.assign(xy=p.pointer_x * 10000 + p.pointer_y)
                           .groupby("cookie_id").xy.nunique() / pg.size())
    edge = (p.pointer_x <= 0) | (p.pointer_y <= 0) | (p.pointer_x >= 1920) | (p.pointer_y >= 1080)
    f["ptr_edge_share"] = edge.groupby(p.cookie_id).mean()
    # шаг курсора между соседними событиями
    step = np.hypot(pg.pointer_x.diff(), pg.pointer_y.diff())
    f["ptr_step_median"] = step.groupby(p.cookie_id).median()
    f["ptr_step0_share"] = (step == 0).groupby(p.cookie_id).mean()
    return f


def ua_features(ev: pd.DataFrame) -> pd.DataFrame:
    """User-Agent разбираем на семейство/версии, а не берём строкой (148 уникальных строк,
    сырая строка переобучается на конкретные версии). Плюс — согласованность UA с platform."""
    ua = ev.drop_duplicates(["cookie_id", "user_agent", "platform"])[["cookie_id", "user_agent", "platform"]].copy()
    s = ua.user_agent.fillna("")
    ua["ua_headless"] = s.str.contains("Headless", case=False)
    ua["ua_script"] = s.str.contains(SCRIPT_UA)
    ua["ua_app"] = s.str.startswith("Avito/")
    ua["ua_mobile"] = s.str.contains("Android|iPhone", case=False)
    ua["ua_os"] = np.select(
        [s.str.contains("Android"), s.str.contains("iPhone|iOS"), s.str.contains("Windows"),
         s.str.contains("Macintosh"), s.str.contains("Linux")],
        ["android", "ios", "windows", "mac", "linux"], "none")
    expected = {"android": "android", "ios": "ios", "windows": "web", "mac": "web", "linux": "web"}
    ua["ua_platform_mismatch"] = ua.ua_os.map(expected).fillna("none") != ua.platform
    ua["chrome_ver"] = s.str.extract(r"Chrome/(\d+)")[0].astype(float)
    ua["app_ver"] = s.str.extract(r"Avito/(\d+)")[0].astype(float)
    ua["os_ver"] = s.str.extract(r"(?:Android|iOS|iPhone OS) (\d+)")[0].astype(float)
    g = ua.groupby("cookie_id")
    f = pd.DataFrame({
        "ua_nunique": g.user_agent.nunique(),
        "ua_headless": g.ua_headless.max().astype(int),
        "ua_script": g.ua_script.max().astype(int),
        "ua_app": g.ua_app.max().astype(int),
        "ua_mobile": g.ua_mobile.max().astype(int),
        "ua_platform_mismatch": g.ua_platform_mismatch.mean(),
        "chrome_ver": g.chrome_ver.max(),
        "app_ver": g.app_ver.max(),
        "os_ver": g.os_ver.max(),
    })
    return f


def meta_features(meta: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=meta.cookie_id.values)
    created = meta.set_index("cookie_id").cookie_created_at
    age = (meta.window_start_ts - meta.cookie_created_at).dt.total_seconds().values / 86400
    f["cookie_age_days"] = age                    # кука всегда создана до начала окна
    first = ev.groupby("cookie_id").event_ts.min()
    f["hours_created_to_first"] = (first - created.reindex(first.index)).dt.total_seconds() / 3600
    return f


# ----------------------------------------------------------------------------- объявления
#
# Сервисы сбора данных обходят общий пул «интересных» им объявлений: ~500 объявлений дают
# 17% просмотров ботов и 2% просмотров людей. Поэтому важно не только КАК кука себя ведёт,
# но и ЧТО она смотрит относительно других кук. Оба признака ниже строго причинные: для
# куки с окном в день D используются только данные, доступные к концу её окна.

def item_pairs(ev: pd.DataFrame) -> pd.DataFrame:
    """Уникальные пары (кука, объявление) с днём окна наблюдения."""
    p = ev.loc[ev.item_id.notna(), ["cookie_id", "item_id", "window_start_ts"]]
    return p.drop_duplicates(["cookie_id", "item_id"]).rename(columns={"window_start_ts": "day"})


def _item_history_sum(pairs: pd.DataFrame, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Для каждой пары (кука, объявление, день D): сумма `values` по ДРУГИМ кукам, смотревшим
    это объявление в дни <= D, и число таких кук. Все окна суточные и заканчиваются в
    полночь, поэтому события других кук того же дня к концу окна уже произошли."""
    days = np.sort(pairs.day.unique())
    d = pairs.day.map(pd.Series(np.arange(len(days)), index=days)).values
    items = pairs.item_id.unique()
    i = pairs.item_id.map(pd.Series(np.arange(len(items)), index=items)).values
    s, n = np.zeros((len(items), len(days))), np.zeros((len(items), len(days)))
    np.add.at(s, (i, d), values)
    np.add.at(n, (i, d), 1)
    return s.cumsum(axis=1)[i, d] - values, n.cumsum(axis=1)[i, d] - 1


def item_popularity_features(pairs: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Без меток: сколько других кук (train и test) смотрели те же объявления в дни <= D.
    Нормируем на 1000 кук истории: для ранних дней история короче, и сырой счётчик был бы
    несопоставим между днями."""
    _, others = _item_history_sum(pairs, np.zeros(len(pairs)))
    day_counts = meta.window_start_ts.value_counts().sort_index().cumsum()
    pop = pd.Series(others / pairs.day.map(day_counts).values * 1000, index=pairs.index)

    g = pop.groupby(pairs.cookie_id)
    return pd.DataFrame({
        "pop_mean": g.mean(),
        "pop_max": g.max(),
        "pop_q75": g.quantile(0.75),
        "pop_hot_share": g.agg(lambda s: (s >= 1.0).mean()),
    })


def item_target_features(pairs: pd.DataFrame, labels: pd.Series, smoothing: float = 5.0) -> pd.DataFrame:
    """С метками: насколько объявления куки «ботовые» по размеченной истории.

    `labels` — cookie_id -> target для кук, метки которых разрешено использовать (train или
    обучающая часть фолда). Куку дня D кодируем только метками кук с днём окна < D, так что
    для train это расширяющееся окно, а для test — вся разметка train. Кука никогда не видит
    свою метку и метки своего дня. Первый день без истории получает NaN.
    """
    p = pairs.assign(t=pairs.cookie_id.map(labels))
    labeled = p[p.t.notna()]
    out = []
    for day in np.sort(p.day.unique()):
        hist = labeled[labeled.day < day]
        if hist.empty:
            continue
        prior = hist.drop_duplicates("cookie_id").t.mean()
        stats = hist.groupby("item_id").t.agg(["sum", "count"])
        q = p[p.day == day].join(stats, on="item_id")
        q[["sum", "count"]] = q[["sum", "count"]].fillna(0)
        q["enc"] = (q["sum"] + prior * smoothing) / (q["count"] + smoothing)
        g = q.groupby("cookie_id")
        out.append(pd.DataFrame({
            "te_mean": g.enc.mean(),
            "te_max": g.enc.max(),
            "te_q75": g.enc.quantile(0.75),
            "te_bot_views": g["sum"].sum(),                           # сколько раз боты уже смотрели
            "te_bot_item_share": g["sum"].agg(lambda s: (s > 0).mean()),
            "te_seen_share": g["count"].agg(lambda s: (s > 0).mean()),
        }))
    return pd.concat(out)


def neighbor_score_features(pairs: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    """Бот-скор «соседей»: средний скор модели первого уровня у других кук, смотревших те же
    объявления в дни <= D. Переносит сигнал на куки без размеченной истории (в т.ч. на
    последние дни теста, до которых метки train не дотягиваются)."""
    s = pairs.cookie_id.map(score).values
    total, n = _item_history_sum(pairs, s)
    q = pd.DataFrame({"mean": np.where(n > 0, total / np.maximum(n, 1), np.nan), "sum": total},
                     index=pairs.index)
    g = q.groupby(pairs.cookie_id)
    return pd.DataFrame({
        "nb_mean": g["mean"].mean(),
        "nb_max": g["mean"].max(),
        "nb_q75": g["mean"].quantile(0.75),
        "nb_sum": g["sum"].sum(),
    })


# ----------------------------------------------------------------------------- сборка

def build_features(events: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Признаки без меток для всех кук из `meta` + пары куки-объявления для target-признаков."""
    ev = clean_events(events, meta)
    pairs = item_pairs(ev)
    parts = [meta_features(meta, ev), timing_features(ev), content_features(ev),
             sequence_features(ev), pointer_features(ev), ua_features(ev),
             item_popularity_features(pairs, meta)]
    X = pd.concat(parts, axis=1).reindex(meta.cookie_id.values)
    X.index.name = "cookie_id"
    return X, pairs


def add_target_features(X: pd.DataFrame, pairs: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    return X.join(item_target_features(pairs, labels))
