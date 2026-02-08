import json
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np
from scipy import stats as sp_stats

from openai import OpenAI

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(
    r'https?://\S+|www\.\S+|lnkd\.in/\S+', re.IGNORECASE
)


def _has_text_link(text: str) -> bool:
    return bool(_URL_PATTERN.search(text or ""))


def _classify_funnel(post: dict) -> str:
    if _has_text_link(post.get("text", "")):
        return "conversion"
    return "awareness"


def _safe_median(values: list) -> float:
    if not values:
        return 0.0
    return float(median(values))


_EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF\U00002600-\U000026FF\U0000FE00-\U0000FE0F"
    "\U0000200D\U00002B50\U0000203C-\U00003299]+",
    re.UNICODE,
)

_CTA_PATTERNS = re.compile(
    r"(coment[áa]|compart[íi]|link en|dej[áa] tu|escribime|descarg[áa]|"
    r"registrate|sum[áa]te|mir[áa]|entr[áa]|hac[ée] click|contame|"
    r"quer[ée]s saber|te interesa|guard[áa]|repost|dale like)",
    re.IGNORECASE,
)

_SPANISH_STOPWORDS = frozenset(
    "de la el en y a los del las un una que es por se con no lo para al "
    "como más pero sus le ya o fue este ha sí porque esta entre cuando muy "
    "sin sobre ser también me hasta desde hay nos durante uno les ni contra "
    "otros ese eso ante ellos e esto mí antes algunos qué unos yo otro "
    "otras otra él tanto esa estos mucho quienes nada muchos cual poco "
    "ella cualquier mi te ti tu tus ellas dos cada cual todas todo toda "
    "los si son así nos u estamos su he sido era donde solo mismo ya mis "
    "tengo somos hace pueden puedo va vamos van este esta estos estas ese "
    "esa esos esas aquel aquella aquellos aquellas ser estar haber tener "
    "hacer poder decir ir ver dar saber querer llegar pasar deber poner "
    "parecer quedar creer hablar llevar dejar seguir encontrar llamar "
    "venir pensar salir volver tomar conocer vivir sentir tratar mirar "
    "contar empezar esperar buscar existir entrar trabajar escribir perder "
    "producir ocurrir entender pedir recibir recordar terminar permitir "
    "aparecer conseguir comenzar".split()
)


def _extract_text_features(post: dict) -> dict:
    """Extract structural features from a post's text."""
    text = post.get("text", "") or ""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    num_paragraphs = max(len(paragraphs), 1)
    avg_paragraph_length = round(sum(len(p) for p in paragraphs) / num_paragraphs, 1) if paragraphs else 0.0

    return {
        "num_mentions": text.count("@"),
        "num_hashtags": text.count("#"),
        "num_paragraphs": num_paragraphs,
        "avg_paragraph_length": avg_paragraph_length,
        "text_length": len(text),
        "has_link": _has_text_link(text),
        "has_emoji": bool(_EMOJI_PATTERN.search(text)),
        "has_cta": bool(_CTA_PATTERNS.search(text)),
    }


def _compute_deep_stats(posts: list[dict]) -> dict:
    """Core statistical engine. Computes everything GPT needs to generate findings.

    Designed to enable "contrarian analysis" — showing where naive dashboard metrics
    are misleading due to outliers, repost contamination, or small sample sizes.
    """
    result: dict = {}

    # ── Engagement array ──
    engagements = np.array([p.get("engagement", 0) for p in posts], dtype=float)

    # ── C1: Outlier detection (IQR method) ──
    outlier_mask = np.zeros(len(engagements), dtype=bool)
    if len(engagements) >= 4:
        q1_val, q3_val = float(np.percentile(engagements, 25)), float(np.percentile(engagements, 75))
        iqr = q3_val - q1_val
        threshold = q3_val + 3 * iqr
        outlier_mask = engagements > threshold
        outlier_indices = [int(i) for i in np.where(outlier_mask)[0]]
        outlier_posts = [
            {
                "date": posts[i]["date"].strftime("%Y-%m-%d %a %H:%M") if hasattr(posts[i].get("date"), "strftime") else str(posts[i].get("date", "")),
                "engagement": int(engagements[i]),
                "content_type": posts[i].get("content_type", "text"),
                "is_repost": posts[i].get("is_repost", False),
                "text": (posts[i].get("text", "") or "")[:150],
            }
            for i in outlier_indices
        ]
        clean_eng = engagements[~outlier_mask]

        result["outliers"] = {
            "threshold": round(threshold, 1),
            "count": int(outlier_mask.sum()),
            "posts": outlier_posts,
            "with_outliers": {
                "mean": round(float(np.mean(engagements)), 1),
                "median": round(float(np.median(engagements)), 1),
                "p25": round(float(np.percentile(engagements, 25)), 1),
                "p75": round(float(np.percentile(engagements, 75)), 1),
                "p90": round(float(np.percentile(engagements, 90)), 1),
            },
            "without_outliers": {
                "mean": round(float(np.mean(clean_eng)), 1) if len(clean_eng) > 0 else 0,
                "median": round(float(np.median(clean_eng)), 1) if len(clean_eng) > 0 else 0,
                "p25": round(float(np.percentile(clean_eng, 25)), 1) if len(clean_eng) > 0 else 0,
                "p75": round(float(np.percentile(clean_eng, 75)), 1) if len(clean_eng) > 0 else 0,
                "p90": round(float(np.percentile(clean_eng, 90)), 1) if len(clean_eng) > 0 else 0,
            },
        }
    else:
        result["outliers"] = {"threshold": 0, "count": 0, "posts": [], "with_outliers": {}, "without_outliers": {}}

    # ── C2: Segment split (reposts vs originals) ──
    originals = [p for p in posts if not p.get("is_repost", False)]
    reposts = [p for p in posts if p.get("is_repost", False)]

    # Repost subcategories: with own comment vs plain reshare
    reposts_with_comment = [p for p in reposts if (p.get("reshare_comment") or "").strip()]
    reposts_plain = [p for p in reposts if not (p.get("reshare_comment") or "").strip()]

    def _seg_stats(post_list):
        if not post_list:
            return {"n": 0, "mean": 0, "median": 0}
        engs = [p.get("engagement", 0) for p in post_list]
        return {
            "n": len(post_list),
            "mean": round(float(np.mean(engs)), 1),
            "median": round(float(np.median(engs)), 1),
        }

    # Mann-Whitney: originals vs reposts
    repost_significance = None
    if len(originals) >= 3 and len(reposts) >= 3:
        orig_engs = [p.get("engagement", 0) for p in originals]
        rep_engs = [p.get("engagement", 0) for p in reposts]
        try:
            stat, pval = sp_stats.mannwhitneyu(orig_engs, rep_engs, alternative="two-sided")
            repost_significance = {"u_stat": round(float(stat), 1), "p_value": round(float(pval), 4)}
        except Exception:
            pass

    # Repost stats without outliers
    originals_no_outlier = [p for i, p in enumerate(posts)
                           if not p.get("is_repost", False) and not outlier_mask[i]]
    reposts_no_outlier = [p for i, p in enumerate(posts)
                         if p.get("is_repost", False) and not outlier_mask[i]]
    repost_sig_clean = None
    if len(originals_no_outlier) >= 3 and len(reposts_no_outlier) >= 3:
        try:
            stat, pval = sp_stats.mannwhitneyu(
                [p.get("engagement", 0) for p in originals_no_outlier],
                [p.get("engagement", 0) for p in reposts_no_outlier],
                alternative="two-sided",
            )
            repost_sig_clean = {"u_stat": round(float(stat), 1), "p_value": round(float(pval), 4)}
        except Exception:
            pass

    result["segment"] = {
        "total": len(posts),
        "originals": _seg_stats(originals),
        "reposts": _seg_stats(reposts),
        "reposts_with_comment": _seg_stats(reposts_with_comment),
        "reposts_plain": _seg_stats(reposts_plain),
        "repost_ratio": round(len(reposts) / len(posts), 3) if posts else 0,
        "repost_vs_original_significance": repost_significance,
        "repost_vs_original_clean": repost_sig_clean,
        "originals_no_outlier": _seg_stats(originals_no_outlier),
        "reposts_no_outlier": _seg_stats(reposts_no_outlier),
    }

    # Monthly repost ratio
    monthly_repost: dict[str, dict] = {}
    for p in posts:
        if hasattr(p.get("date"), "strftime"):
            month_key = p["date"].strftime("%Y-%m")
            monthly_repost.setdefault(month_key, {"total": 0, "reposts": 0})
            monthly_repost[month_key]["total"] += 1
            if p.get("is_repost", False):
                monthly_repost[month_key]["reposts"] += 1
    result["segment"]["monthly_repost_ratio"] = {
        m: round(d["reposts"] / d["total"], 3) if d["total"] else 0
        for m, d in sorted(monthly_repost.items())
    }

    # All subsequent analysis on ORIGINALS ONLY (cleaner signal)
    analysis_posts = originals if originals else posts

    # ── C3: Per-format comparison (originals only, WITH and WITHOUT outliers) ──
    def _format_breakdown(post_list):
        groups: dict[str, list[float]] = {}
        for p in post_list:
            ct = p.get("content_type", "text")
            groups.setdefault(ct, []).append(p.get("engagement", 0))
        stats = {}
        for ct, engs in groups.items():
            arr = np.array(engs, dtype=float)
            stats[ct] = {
                "n": len(engs),
                "median": round(float(np.median(arr)), 1),
                "p25": round(float(np.percentile(arr, 25)), 1),
                "p75": round(float(np.percentile(arr, 75)), 1),
                "mean": round(float(np.mean(arr)), 1),
            }
        return stats, groups

    format_stats, format_groups = _format_breakdown(analysis_posts)

    # Also compute format stats for ALL posts (including reposts) to show contamination
    format_stats_all, _ = _format_breakdown(posts)

    # Format stats WITHOUT outliers (originals only)
    analysis_no_outlier = [p for i, p in enumerate(posts)
                          if not p.get("is_repost", False) and not outlier_mask[i]]
    format_stats_clean, _ = _format_breakdown(analysis_no_outlier)

    # Mann-Whitney U between format pairs (originals only)
    format_comparisons = []
    format_names = list(format_groups.keys())
    for i in range(len(format_names)):
        for j in range(i + 1, len(format_names)):
            a_name, b_name = format_names[i], format_names[j]
            a_vals, b_vals = format_groups[a_name], format_groups[b_name]
            if len(a_vals) >= 3 and len(b_vals) >= 3:
                try:
                    stat, pval = sp_stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
                    format_comparisons.append({
                        "a": a_name, "b": b_name,
                        "u_stat": round(float(stat), 1),
                        "p_value": round(float(pval), 4),
                        "significant": bool(pval < 0.05),
                        "higher": a_name if float(np.median(a_vals)) > float(np.median(b_vals)) else b_name,
                    })
                except Exception:
                    pass

    result["format_stats"] = format_stats
    result["format_stats_all"] = format_stats_all
    result["format_stats_clean"] = format_stats_clean
    result["format_comparisons"] = format_comparisons

    # ── C4: Spearman correlations (originals only) ──
    correlations = []
    eng_array = np.array([p.get("engagement", 0) for p in analysis_posts], dtype=float)

    if len(analysis_posts) >= 5:
        features_list = [_extract_text_features(p) for p in analysis_posts]
        numeric_features = {
            "text_length": [f["text_length"] for f in features_list],
            "num_mentions": [f["num_mentions"] for f in features_list],
            "num_hashtags": [f["num_hashtags"] for f in features_list],
            "num_paragraphs": [f["num_paragraphs"] for f in features_list],
            "avg_paragraph_length": [f["avg_paragraph_length"] for f in features_list],
            "has_link": [int(f["has_link"]) for f in features_list],
            "has_emoji": [int(f["has_emoji"]) for f in features_list],
            "has_cta": [int(f["has_cta"]) for f in features_list],
            "hour_posted": [
                p["date"].hour if hasattr(p.get("date"), "hour") else 0
                for p in analysis_posts
            ],
            "day_of_week": [
                p["date"].weekday() if hasattr(p.get("date"), "weekday") else 0
                for p in analysis_posts
            ],
        }

        for feat_name, feat_vals in numeric_features.items():
            feat_array = np.array(feat_vals, dtype=float)
            if np.std(feat_array) == 0:
                continue
            try:
                rho, pval = sp_stats.spearmanr(feat_array, eng_array)
                if not np.isnan(rho):
                    correlations.append({
                        "feature": feat_name,
                        "rho": round(float(rho), 3),
                        "p_value": round(float(pval), 4),
                        "significant": bool(pval < 0.05),
                        "direction": "positive" if rho > 0 else "negative",
                    })
            except Exception:
                pass

    result["correlations"] = sorted(correlations, key=lambda x: abs(x["rho"]), reverse=True)

    # ── C5: Text length buckets (finer grained, with/without outlier) ──
    length_buckets_def = [
        ("0-100", 0, 100), ("100-300", 100, 300), ("300-500", 300, 500),
        ("500-800", 500, 800), ("800-1K", 800, 1000), ("1K-1.5K", 1000, 1500),
        ("1.5K-3K", 1500, 3000), ("3K+", 3000, 999999),
    ]

    def _bucket_stats(post_list):
        buckets = {}
        for label, lo, hi in length_buckets_def:
            bucket_posts = [p for p in post_list
                          if lo <= len(p.get("text", "") or "") < hi]
            if bucket_posts:
                engs = [p.get("engagement", 0) for p in bucket_posts]
                buckets[label] = {
                    "n": len(bucket_posts),
                    "mean": round(float(np.mean(engs)), 1),
                    "median": round(float(np.median(engs)), 1),
                }
        return buckets

    result["length_buckets_originals"] = _bucket_stats(analysis_posts)
    result["length_buckets_clean"] = _bucket_stats(analysis_no_outlier)

    # ── C5b: Link penalty analysis ──
    with_links = [p for p in analysis_posts if _has_text_link(p.get("text", "") or "")]
    without_links = [p for p in analysis_posts if not _has_text_link(p.get("text", "") or "")]
    result["link_analysis"] = {
        "with_links": _seg_stats(with_links),
        "without_links": _seg_stats(without_links),
    }
    if len(with_links) >= 3 and len(without_links) >= 3:
        try:
            stat, pval = sp_stats.mannwhitneyu(
                [p.get("engagement", 0) for p in with_links],
                [p.get("engagement", 0) for p in without_links],
                alternative="two-sided",
            )
            result["link_analysis"]["significance"] = {"u_stat": round(float(stat), 1), "p_value": round(float(pval), 4)}
        except Exception:
            pass

    # ── C5c: Mention buckets ──
    def _mention_count(p):
        return (p.get("text", "") or "").count("@")

    mention_buckets = {"0": [], "1-2": [], "3-4": [], "5+": []}
    for p in analysis_posts:
        mc = _mention_count(p)
        if mc == 0:
            mention_buckets["0"].append(p.get("engagement", 0))
        elif mc <= 2:
            mention_buckets["1-2"].append(p.get("engagement", 0))
        elif mc <= 4:
            mention_buckets["3-4"].append(p.get("engagement", 0))
        else:
            mention_buckets["5+"].append(p.get("engagement", 0))

    result["mention_buckets"] = {
        k: {"n": len(v), "median": round(float(np.median(v)), 1) if v else 0, "mean": round(float(np.mean(v)), 1) if v else 0}
        for k, v in mention_buckets.items() if v
    }

    # ── C6: Keyword analysis (top vs bottom quartile) ──
    keyword_analysis: dict = {}
    if len(analysis_posts) >= 8:
        sorted_by_eng = sorted(analysis_posts, key=lambda x: x.get("engagement", 0))
        q_size = max(len(sorted_by_eng) // 4, 1)
        bottom_q = sorted_by_eng[:q_size]
        top_q = sorted_by_eng[-q_size:]

        def _tokenize(posts_list: list[dict]) -> Counter:
            counter: Counter = Counter()
            for p in posts_list:
                text = (p.get("text", "") or "").lower()
                words = re.findall(r"[a-záéíóúñü]{3,}", text)
                for w in words:
                    if w not in _SPANISH_STOPWORDS and len(w) >= 3:
                        counter[w] += 1
            return counter

        top_words = _tokenize(top_q)
        bottom_words = _tokenize(bottom_q)

        top_norm = {w: c / len(top_q) for w, c in top_words.items() if c >= 2}
        bottom_norm = {w: c / len(bottom_q) for w, c in bottom_words.items() if c >= 2}

        top_dominant = []
        for w, freq in top_norm.items():
            bottom_freq = bottom_norm.get(w, 0.01)
            ratio = freq / bottom_freq
            if ratio >= 2:
                top_dominant.append({"word": w, "top_freq": round(freq, 2), "bottom_freq": round(bottom_freq, 2), "ratio": round(ratio, 1)})
        top_dominant.sort(key=lambda x: x["ratio"], reverse=True)

        bottom_dominant = []
        for w, freq in bottom_norm.items():
            top_freq = top_norm.get(w, 0.01)
            ratio = freq / top_freq
            if ratio >= 2:
                bottom_dominant.append({"word": w, "bottom_freq": round(freq, 2), "top_freq": round(top_freq, 2), "ratio": round(ratio, 1)})
        bottom_dominant.sort(key=lambda x: x["ratio"], reverse=True)

        keyword_analysis = {
            "top_quartile_size": len(top_q),
            "bottom_quartile_size": len(bottom_q),
            "top_quartile_words": top_dominant[:15],
            "bottom_quartile_words": bottom_dominant[:15],
        }

    result["keyword_analysis"] = keyword_analysis

    # ── C7: Temporal trend ──
    temporal: dict = {}

    # Monthly stats (median + mean + post count)
    monthly_data: dict[str, list] = {}
    for p in analysis_posts:
        if hasattr(p.get("date"), "strftime"):
            mk = p["date"].strftime("%Y-%m")
            monthly_data.setdefault(mk, []).append(p.get("engagement", 0))

    monthly_stats = {}
    for k in sorted(monthly_data.keys()):
        v = monthly_data[k]
        monthly_stats[k] = {
            "n": len(v),
            "median": round(float(np.median(v)), 1),
            "mean": round(float(np.mean(v)), 1),
        }
    temporal["monthly_stats"] = monthly_stats

    # Also compute monthly stats for impressions
    monthly_imp: dict[str, list] = {}
    for p in analysis_posts:
        if hasattr(p.get("date"), "strftime") and p.get("impressions", 0) > 0:
            mk = p["date"].strftime("%Y-%m")
            monthly_imp.setdefault(mk, []).append(p.get("impressions", 0))
    temporal["monthly_impressions"] = {
        k: {"n": len(v), "mean": round(float(np.mean(v)), 1)}
        for k, v in sorted(monthly_imp.items())
    }

    medians_list = [monthly_stats[k]["median"] for k in sorted(monthly_stats.keys())]
    months_list = sorted(monthly_stats.keys())

    if len(medians_list) >= 3:
        try:
            tau, pval = sp_stats.kendalltau(range(len(medians_list)), medians_list)
            temporal["kendall_tau"] = round(float(tau), 3)
            temporal["kendall_p"] = round(float(pval), 4)
            temporal["direction"] = "growing" if float(tau) > 0 and float(pval) < 0.05 else "declining" if float(tau) < 0 and float(pval) < 0.05 else "stable"
        except Exception:
            temporal["direction"] = "unknown"

        # Last month vs historical
        last_month_val = medians_list[-1]
        historical_vals = medians_list[:-1]
        if historical_vals:
            hist_median = float(np.median(historical_vals))
            pct_change = round((last_month_val - hist_median) / hist_median * 100, 1) if hist_median > 0 else 0.0
            temporal["last_month"] = months_list[-1]
            temporal["last_month_median"] = last_month_val
            temporal["historical_median"] = round(hist_median, 1)
            temporal["pct_change"] = pct_change

    # Kruskal-Wallis across weekdays
    weekday_groups: dict[int, list[float]] = {}
    for p in analysis_posts:
        if hasattr(p.get("date"), "weekday"):
            wd = p["date"].weekday()
            weekday_groups.setdefault(wd, []).append(p.get("engagement", 0))

    if len(weekday_groups) >= 3:
        groups_with_data = [v for v in weekday_groups.values() if len(v) >= 2]
        if len(groups_with_data) >= 3:
            try:
                h_stat, pval = sp_stats.kruskal(*groups_with_data)
                temporal["kruskal_wallis_h"] = round(float(h_stat), 2)
                temporal["kruskal_wallis_p"] = round(float(pval), 4)
                temporal["day_significant"] = bool(pval < 0.05)
            except Exception:
                temporal["day_significant"] = False
        else:
            temporal["day_significant"] = False
    else:
        temporal["day_significant"] = False

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    temporal["weekday_medians"] = {
        day_names[wd]: round(float(np.median(vals)), 1)
        for wd, vals in sorted(weekday_groups.items())
        if vals
    }

    # Hour analysis WITH vs WITHOUT outliers
    hour_groups: dict[int, list[float]] = {}
    hour_groups_clean: dict[int, list[float]] = {}
    for i, p in enumerate(posts):
        if hasattr(p.get("date"), "hour"):
            h = p["date"].hour
            hour_groups.setdefault(h, []).append(p.get("engagement", 0))
            if not outlier_mask[i]:
                hour_groups_clean.setdefault(h, []).append(p.get("engagement", 0))
    temporal["hour_avg_with_outliers"] = {
        h: round(float(np.mean(v)), 1) for h, v in sorted(hour_groups.items()) if v
    }
    temporal["hour_avg_without_outliers"] = {
        h: round(float(np.mean(v)), 1) for h, v in sorted(hour_groups_clean.items()) if v
    }

    result["temporal"] = temporal

    # ── C8: Impression efficiency ──
    impression_analysis: dict = {}
    posts_with_imp = [p for p in analysis_posts if p.get("impressions", 0) > 0]
    if posts_with_imp:
        rates = []
        high_reach_low_eng = []
        low_reach_high_eng = []

        for p in posts_with_imp:
            imp = p.get("impressions", 0)
            eng = p.get("engagement", 0)
            rate = eng / imp * 100 if imp > 0 else 0
            rates.append(rate)

            post_summary = {
                "date": p["date"].strftime("%Y-%m-%d") if hasattr(p.get("date"), "strftime") else str(p.get("date", "")),
                "engagement": eng,
                "impressions": imp,
                "rate": round(rate, 2),
                "content_type": p.get("content_type", "text"),
                "text": (p.get("text", "") or "")[:120],
            }

            if imp > 2000 and rate < 1.5:
                high_reach_low_eng.append(post_summary)
            if rate > 4:
                low_reach_high_eng.append(post_summary)

        rates_arr = np.array(rates, dtype=float)
        impression_analysis = {
            "total_with_impressions": len(posts_with_imp),
            "median_rate": round(float(np.median(rates_arr)), 2),
            "mean_rate": round(float(np.mean(rates_arr)), 2),
            "high_reach_low_engagement": sorted(high_reach_low_eng, key=lambda x: x["impressions"], reverse=True)[:5],
            "low_reach_high_engagement": sorted(low_reach_high_eng, key=lambda x: x["rate"], reverse=True)[:5],
        }

    result["impression_efficiency"] = impression_analysis

    # ── C9: Comment/reaction ratio by format ──
    comment_ratio_by_type: dict = {}
    for p in analysis_posts:
        ct = p.get("content_type", "text")
        reactions = p.get("reactions", 0)
        comments = p.get("comments", 0)
        comment_ratio_by_type.setdefault(ct, {"reactions": 0, "comments": 0})
        comment_ratio_by_type[ct]["reactions"] += reactions
        comment_ratio_by_type[ct]["comments"] += comments
    result["comment_ratio_by_type"] = {
        ct: round(d["comments"] / d["reactions"], 3) if d["reactions"] > 0 else 0
        for ct, d in comment_ratio_by_type.items()
    }

    # ── C10: P90 threshold for "top 10%" ──
    if len(analysis_posts) >= 10:
        orig_engs = np.array([p.get("engagement", 0) for p in analysis_posts], dtype=float)
        result["top_10_threshold"] = round(float(np.percentile(orig_engs, 90)), 1)
    else:
        result["top_10_threshold"] = 0

    return result


ANALYSIS_DIR = Path("data/analysis")
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

DASHBOARD_CACHE_PATH = ANALYSIS_DIR / "dashboard_cache.json"

# --- Per-post classification (cheap, gpt-4o-mini) ---

CLASSIFY_PROMPT = """Analyze this LinkedIn post and return a JSON object with:
- "category": one of ["thought_leadership", "personal_story", "industry_news", "how_to", "promotion", "engagement_bait", "announcement", "case_study", "other"]
- "sentiment": one of ["positive", "neutral", "negative", "inspirational", "controversial"]
- "topics": list of 1-3 main topics/keywords (short phrases in Spanish)
- "image_type": if the post has an image, classify as one of ["infographic", "personal_photo", "screenshot", "carousel", "meme", "quote_card", "chart_data", "product", "event", "none"]. Use "none" if no image.

Post text:
{text}

Has image: {has_image}
Content type: {content_type}

Return ONLY valid JSON, no markdown formatting."""

# --- Deep pattern analysis (expensive, gpt-4o, one call with all data) ---

PATTERN_ANALYSIS_PROMPT = """Sos un analista experto de LinkedIn. Tu trabajo es encontrar HALLAZGOS que contradigan las conclusiones ingenuas del dashboard. El dashboard calcula promedios simples contaminados por outliers, mezcla reposts con originales, y no hace tests estadísticos. Vos tenés las estadísticas reales.

TU ROL: Actuar como un "fact-checker" del dashboard. Cada hallazgo debe mostrar qué dice el dashboard (métricas ingenuas) vs qué dice la realidad (datos limpios con tests).

REGLAS:
1. SIEMPRE usá medianas (no promedios) salvo para mostrar cómo los promedios mienten
2. Si avg >> mediana, señalalo explícitamente como sesgo por outlier
3. Citá p-values, rho, n de muestra en cada afirmación
4. Si una diferencia NO es significativa (p > 0.05), decilo — es tan valioso como una que sí lo es
5. Escribí en español argentino (vos, usá, poné, etc.)
6. Sé provocativo y directo. No des consejos genéricos de LinkedIn.

DATOS DEL PERFIL:
- Total posts: {total_posts}
- Período: {date_range}
- Segmentación: {segment_stats}

DETECCIÓN DE OUTLIERS (IQR * 3):
{outlier_stats}

FORMATO — Dashboard (todos los posts) vs Realidad (solo originales) vs Limpio (sin outliers):
Dashboard: {format_stats_all}
Originales: {format_stats}
Limpios: {format_stats_clean}

MANN-WHITNEY ENTRE FORMATOS:
{format_comparisons}

CORRELACIONES SPEARMAN (todas, incluso no significativas):
{correlations}

BUCKETS DE LARGO DE TEXTO (originales sin outlier):
{length_buckets}

ANÁLISIS DE LINKS:
{link_analysis}

MENCIONES POR BUCKET:
{mention_buckets}

KEYWORDS (cuartil superior vs inferior):
{keyword_analysis}

TENDENCIA TEMPORAL:
{temporal_stats}

EFICIENCIA DE IMPRESIONES:
{impression_stats}

RATIO COMENTARIOS/REACCIONES POR TIPO:
{comment_ratios}

UMBRAL TOP 10% (P90 originales): {top_10_threshold}

TOP 10 POSTS:
{top_posts}

BOTTOM 10 POSTS:
{bottom_posts}

Devolvé un JSON con esta estructura:
{{
  "findings": [
    {{
      "id": "identificador_unico",
      "title": "Título provocativo que resuma el hallazgo (con datos clave)",
      "severity": "critical|warning|insight",
      "summary": "1-2 oraciones con el hallazgo principal y el número más impactante",
      "details": [
        "Detalle 1 con números específicos y comparación dashboard vs realidad",
        "Detalle 2 con evidencia estadística (rho, p-value, medianas)",
        "Detalle 3 con contexto o implicancias",
        "Detalle 4 si aplica"
      ],
      "action": "Acción concreta y específica basada en este hallazgo"
    }}
  ],
  "key_metrics": [
    {{
      "label": "Nombre corto de la métrica",
      "value": "Valor real (mediana, limpio)",
      "subtext": "Por qué el dashboard dice otra cosa"
    }}
  ],
  "dashboard_vs_reality": [
    {{
      "dashboard_says": "Lo que el dashboard muestra (con número)",
      "reality": "Lo que los datos limpios dicen (con número)"
    }}
  ],
  "optimal_formula": {{
    "format": "Formato óptimo con evidencia",
    "length": "Rango de caracteres óptimo con evidencia",
    "mentions": "Cantidad de menciones con evidencia",
    "links": "Estrategia de links con evidencia",
    "content_style": "Estilo/tono con evidencia de keywords",
    "hashtags": "Estrategia de hashtags con evidencia",
    "when": "Día/hora con evidencia (o si NO importa, decirlo)",
    "repost_limit": "Ratio máximo recomendado de reposts"
  }},
  "top_3_actions": [
    {{
      "action": "Acción específica y accionable",
      "expected_impact": "Impacto esperado con números de los datos"
    }}
  ]
}}

IMPORTANTE:
- Generá MÍNIMO 6 findings. Cada uno debe tener datos específicos, no generalidades.
- severity "critical" = el dashboard miente o hay un problema grave
- severity "warning" = algo que se puede mejorar con datos
- severity "insight" = patrón interesante que no es obvio
- Los findings deben cubrir: outliers, reposts, formato real, largo, menciones, links, keywords, tendencia temporal, impressiones desperdiciadas
- dashboard_vs_reality debe tener al menos 4 items mostrando métricas que cambian al limpiar datos

Return ONLY valid JSON, no markdown."""


def _get_analysis_cache_path() -> Path:
    return ANALYSIS_DIR / "analysis_cache.json"


def _load_analysis_cache() -> dict[str, dict]:
    path = _get_analysis_cache_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _save_analysis_cache(cache: dict[str, dict]):
    path = _get_analysis_cache_path()
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _post_cache_key(post: dict) -> str:
    text = post.get("text", "")[:80]
    ts = post.get("timestamp", 0)
    if not ts and post.get("date"):
        d = post["date"]
        if hasattr(d, "timestamp"):
            ts = int(d.timestamp())
    return f"{text}|{ts}"


def save_dashboard(metrics: dict):
    """Save dashboard metrics to disk for later reload."""
    DASHBOARD_CACHE_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Dashboard saved to {DASHBOARD_CACHE_PATH}")


def load_dashboard() -> dict | None:
    """Load previously saved dashboard metrics."""
    if DASHBOARD_CACHE_PATH.exists():
        try:
            data = json.loads(DASHBOARD_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "total_posts" in data:
                return data
        except Exception:
            pass
    return None


def classify_posts(posts: list[dict], openai_api_key: str, progress_callback=None) -> list[dict]:
    """Phase 1: Classify each post (category, sentiment, topics). Uses cache."""
    client = OpenAI(api_key=openai_api_key)
    cache = _load_analysis_cache()
    analyzed = []
    total = len(posts)
    cached_count = 0
    api_count = 0

    for i, post in enumerate(posts):
        key = _post_cache_key(post)

        if key in cache:
            cached = cache[key]
            post["category"] = cached.get("category", "other")
            post["sentiment"] = cached.get("sentiment", "neutral")
            post["topics"] = cached.get("topics", [])
            post["image_type"] = cached.get("image_type", "none")
            analyzed.append(post)
            cached_count += 1
            if progress_callback:
                progress_callback(i + 1, total, from_cache=True)
            continue

        if not post.get("text"):
            result = {"category": "other", "sentiment": "neutral", "topics": [], "image_type": "none"}
        else:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(
                        text=post["text"][:1500],
                        has_image=post.get("has_image", False),
                        content_type=post.get("content_type", "text"),
                    )}],
                    temperature=0.2,
                    max_tokens=250,
                )
                raw = response.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
                result = json.loads(raw)
                api_count += 1
            except Exception as e:
                logger.warning(f"Classification failed for post {i+1}/{total}: {e}")
                result = {"category": "other", "sentiment": "neutral", "topics": [], "image_type": "none"}

        post["category"] = result.get("category", "other")
        post["sentiment"] = result.get("sentiment", "neutral")
        post["topics"] = result.get("topics", [])
        post["image_type"] = result.get("image_type", "none")
        analyzed.append(post)

        cache[key] = {
            "category": post["category"],
            "sentiment": post["sentiment"],
            "topics": post["topics"],
            "image_type": post["image_type"],
        }

        if (i + 1) % 5 == 0:
            _save_analysis_cache(cache)
            logger.info(f"Classification: {i+1}/{total} (API: {api_count}, cached: {cached_count})")

        if progress_callback:
            progress_callback(i + 1, total, from_cache=False)

    _save_analysis_cache(cache)
    logger.info(f"Classification done: {total} posts (API: {api_count}, cached: {cached_count})")
    return analyzed


def deep_pattern_analysis(posts: list[dict], metrics: dict, openai_api_key: str) -> dict:
    """Phase 2: Compute statistics in Python, send to GPT for interpretation.

    Note: File-based caching removed for multi-tenant safety.
    Caching is handled at the DB level in dashboard.py (per user_id + date range).
    """
    client = OpenAI(api_key=openai_api_key)

    # ── Compute all statistics in Python ──
    logger.info("Computing deep statistics...")
    deep_stats = _compute_deep_stats(posts)

    # ── Prepare post summaries (top/bottom 10) ──
    sorted_posts = sorted(posts, key=lambda x: x.get("engagement", 0), reverse=True)

    def post_summary(p, rank=None):
        prefix = f"#{rank} " if rank else ""
        date = p["date"].strftime("%Y-%m-%d %a %H:%M") if hasattr(p.get("date"), "strftime") else str(p.get("date", ""))
        repost_tag = " [REPOST]" if p.get("is_repost") else ""
        imp = p.get("impressions", 0)
        imp_str = f" imp={imp}" if imp > 0 else ""
        return (
            f"{prefix}[{date}] eng={p.get('engagement',0)} (react={p.get('reactions',0)} "
            f"comm={p.get('comments',0)}{imp_str}){repost_tag} "
            f"type={p.get('content_type','text')} cat={p.get('category','?')} "
            f"len={len(p.get('text',''))} chars\n"
            f"  \"{p.get('text','')[:150]}\""
        )

    top_posts = "\n\n".join(post_summary(p, i+1) for i, p in enumerate(sorted_posts[:10]))
    bottom_posts = "\n\n".join(post_summary(p, i+1) for i, p in enumerate(sorted_posts[-10:]))

    # Date range
    dates = [p["date"] for p in posts if hasattr(p.get("date"), "strftime")]
    date_range = ""
    if dates:
        date_range = f"{min(dates).strftime('%Y-%m-%d')} a {max(dates).strftime('%Y-%m-%d')}"

    # Format stats for prompt
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    def _fmt_json(obj):
        return json.dumps(obj, ensure_ascii=False, indent=2, cls=_NumpyEncoder)

    prompt = PATTERN_ANALYSIS_PROMPT.format(
        total_posts=len(posts),
        date_range=date_range,
        segment_stats=_fmt_json(deep_stats.get("segment", {})),
        outlier_stats=_fmt_json(deep_stats.get("outliers", {})),
        format_stats_all=_fmt_json(deep_stats.get("format_stats_all", {})),
        format_stats=_fmt_json(deep_stats.get("format_stats", {})),
        format_stats_clean=_fmt_json(deep_stats.get("format_stats_clean", {})),
        format_comparisons=_fmt_json(deep_stats.get("format_comparisons", [])),
        correlations=_fmt_json(deep_stats.get("correlations", [])),
        length_buckets=_fmt_json(deep_stats.get("length_buckets_clean", {})),
        link_analysis=_fmt_json(deep_stats.get("link_analysis", {})),
        mention_buckets=_fmt_json(deep_stats.get("mention_buckets", {})),
        keyword_analysis=_fmt_json(deep_stats.get("keyword_analysis", {})),
        temporal_stats=_fmt_json(deep_stats.get("temporal", {})),
        impression_stats=_fmt_json(deep_stats.get("impression_efficiency", {})),
        comment_ratios=_fmt_json(deep_stats.get("comment_ratio_by_type", {})),
        top_10_threshold=deep_stats.get("top_10_threshold", 0),
        top_posts=top_posts,
        bottom_posts=bottom_posts,
    )

    logger.info(f"Pattern analysis prompt: {len(prompt)} chars")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=6000,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        analysis = json.loads(raw)

        # Attach raw stats so frontend can display them directly
        analysis["_raw_stats"] = deep_stats

        logger.info("Pattern analysis complete")
        return analysis

    except Exception as e:
        logger.error(f"Pattern analysis failed: {e}", exc_info=True)
        return {"error": str(e)}


def compute_metrics(posts: list[dict]) -> dict:
    """Compute all dashboard metrics from analyzed posts."""
    if not posts:
        return {}

    # ── Filter out reposts: use only original posts for metrics ──
    all_posts = posts
    originals = [p for p in posts if not p.get("is_repost", False)]
    reposts = [p for p in posts if p.get("is_repost", False)]
    posts = originals if originals else all_posts

    has_engagement = any(p.get("engagement", 0) > 0 for p in posts)

    posts_by_month = Counter()
    engagement_by_month = Counter()
    impressions_by_month = Counter()
    month_counts = Counter()
    for p in posts:
        key = p["date"].strftime("%Y-%m")
        posts_by_month[key] += 1
        engagement_by_month[key] += p.get("engagement", 0)
        impressions_by_month[key] += p.get("impressions", 0)
        month_counts[key] += 1
    sorted_months = sorted(posts_by_month.keys())

    avg_engagement_by_month = {
        m: round(engagement_by_month[m] / month_counts[m], 1) if month_counts[m] else 0
        for m in sorted_months
    }
    avg_impressions_by_month = {
        m: round(impressions_by_month[m] / month_counts[m], 1) if month_counts[m] else 0
        for m in sorted_months
    }

    posts_by_week = Counter()
    for p in posts:
        posts_by_week[p["date"].strftime("%Y-W%V")] += 1
    sorted_weeks = sorted(posts_by_week.keys())

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    posts_by_day = Counter()
    engagement_by_day = Counter()
    day_counts = Counter()
    for p in posts:
        d = p["date"].weekday()
        posts_by_day[d] += 1
        engagement_by_day[d] += p.get("engagement", 0)
        day_counts[d] += 1

    posts_by_hour = Counter()
    engagement_by_hour = Counter()
    hour_counts = Counter()
    for p in posts:
        h = p["date"].hour
        posts_by_hour[h] += 1
        engagement_by_hour[h] += p.get("engagement", 0)
        hour_counts[h] += 1

    content_types = Counter(p.get("content_type", "text") for p in posts)

    eng_by_type = {}
    type_counts = Counter()
    for p in posts:
        ct = p.get("content_type", "text")
        eng_by_type[ct] = eng_by_type.get(ct, 0) + p.get("engagement", 0)
        type_counts[ct] += 1
    avg_eng_by_type = {
        ct: round(eng_by_type[ct] / type_counts[ct], 1) for ct in eng_by_type
    }

    # All posts data for drill-down filtering from charts
    # Include ALL posts (originals + reposts) so UI can show/filter them
    all_posts_data = [
        {
            "text": p["text"][:200] + ("..." if len(p["text"]) > 200 else ""),
            "date": p["date"].strftime("%Y-%m-%d"),
            "day": p["date"].strftime("%a"),
            "reactions": p.get("reactions", 0),
            "comments": p.get("comments", 0),
            "shares": p.get("shares", 0),
            "engagement": p.get("engagement", 0),
            "impressions": p.get("impressions", 0),
            "content_type": p.get("content_type", "text"),
            "category": p.get("category", ""),
            "sentiment": p.get("sentiment", "neutral"),
            "has_image": p.get("has_image", False),
            "text_length": p.get("text_length", len(p.get("text", ""))),
            "url": p.get("url", ""),
            "topics": p.get("topics", []),
            "is_repost": p.get("is_repost", False),
            "original_author": p.get("original_author", ""),
        }
        for p in all_posts
    ]

    top_posts = sorted(posts, key=lambda x: x.get("engagement", 0), reverse=True)[:20]
    top_posts_data = [
        {
            "text": p["text"][:200] + ("..." if len(p["text"]) > 200 else ""),
            "date": p["date"].strftime("%Y-%m-%d"),
            "reactions": p.get("reactions", 0),
            "comments": p.get("comments", 0),
            "shares": p.get("shares", 0),
            "engagement": p.get("engagement", 0),
            "impressions": p.get("impressions", 0),
            "content_type": p.get("content_type", "text"),
            "category": p.get("category", ""),
            "url": p.get("url", ""),
            "is_repost": p.get("is_repost", False),
            "original_author": p.get("original_author", ""),
        }
        for p in top_posts
    ]

    with_img = [p for p in posts if p.get("has_image")]
    without_img = [p for p in posts if not p.get("has_image")]
    avg_eng_with = round(sum(p.get("engagement", 0) for p in with_img) / len(with_img), 1) if with_img else 0
    avg_eng_without = round(sum(p.get("engagement", 0) for p in without_img) / len(without_img), 1) if without_img else 0

    image_types = Counter(p.get("image_type", "none") for p in posts if p.get("image_type") != "none")
    categories = Counter(p.get("category", "other") for p in posts)
    sentiments = Counter(p.get("sentiment", "neutral") for p in posts)

    all_topics = []
    for p in posts:
        all_topics.extend(p.get("topics", []))
    top_keywords = Counter(all_topics).most_common(30)

    buckets = {"0-100": 0, "100-300": 0, "300-500": 0, "500-1000": 0, "1000+": 0}
    for p in posts:
        l = p.get("text_length", len(p.get("text", "")))
        if l < 100: buckets["0-100"] += 1
        elif l < 300: buckets["100-300"] += 1
        elif l < 500: buckets["300-500"] += 1
        elif l < 1000: buckets["500-1000"] += 1
        else: buckets["1000+"] += 1

    scatter_data = [
        {"x": p.get("text_length", len(p.get("text", ""))), "y": p.get("engagement", 0)}
        for p in posts
    ]

    # ── A. Funnel Analysis ──
    funnel_data = {}
    for p in posts:
        stage = _classify_funnel(p)
        funnel_data.setdefault(stage, []).append(p)

    funnel_analysis = {}
    for stage, stage_posts in funnel_data.items():
        engs = [p.get("engagement", 0) for p in stage_posts]
        reactions = sum(p.get("reactions", 0) for p in stage_posts)
        comments = sum(p.get("comments", 0) for p in stage_posts)
        imps = [p.get("impressions", 0) for p in stage_posts]
        funnel_analysis[stage] = {
            "count": len(stage_posts),
            "median_engagement": round(_safe_median(engs), 1),
            "conversation_ratio": round(comments / reactions, 3) if reactions else 0,
            "avg_impressions": round(sum(imps) / len(imps), 1) if imps else 0,
        }

    # ── B. Topic Matrix ──
    topic_groups: dict[str, list] = {}
    for p in posts:
        topics_list = p.get("topics", [])
        if topics_list:
            topic = topics_list[0]
            topic_groups.setdefault(topic, []).append(p)

    topic_matrix = []
    for topic, tposts in topic_groups.items():
        engs = [p.get("engagement", 0) for p in tposts]
        imps = [p.get("impressions", 0) for p in tposts]
        topic_matrix.append({
            "topic": topic,
            "count": len(tposts),
            "median_engagement": round(_safe_median(engs), 1),
            "avg_impressions": round(sum(imps) / len(imps), 1) if imps else 0,
        })
    topic_matrix.sort(key=lambda x: x["median_engagement"], reverse=True)
    topic_matrix = topic_matrix[:8]

    # ── C. Day x Hour Heatmap ──
    heatmap_eng: dict[tuple[int, int], list] = {}
    for p in posts:
        key = (p["date"].weekday(), p["date"].hour)
        heatmap_eng.setdefault(key, []).append(p.get("engagement", 0))

    heatmap_data = []
    heatmap_max = 0
    for wd in range(7):
        for hr in range(24):
            vals = heatmap_eng.get((wd, hr), [])
            avg = round(sum(vals) / len(vals), 1) if vals else 0
            heatmap_data.append({"day": wd, "hour": hr, "value": avg})
            if avg > heatmap_max:
                heatmap_max = avg

    day_hour_heatmap = {
        "data": heatmap_data,
        "max_value": heatmap_max,
        "day_labels": ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"],
    }

    # ── D. Strategic Cards ──
    total_impressions = sum(p.get("impressions", 0) for p in posts)
    awareness_posts = funnel_data.get("awareness", [])
    conversion_posts = funnel_data.get("conversion", [])

    awareness_engs = [p.get("engagement", 0) for p in awareness_posts]
    avg_eng_no_link = round(sum(awareness_engs) / len(awareness_engs), 1) if awareness_engs else 0

    conv_reactions = sum(p.get("reactions", 0) for p in conversion_posts)
    conv_comments = sum(p.get("comments", 0) for p in conversion_posts)
    conv_ratio = round(conv_comments / conv_reactions, 3) if conv_reactions else 0

    best_topic = topic_matrix[0] if topic_matrix else {"topic": "N/A", "median_engagement": 0}
    tl_count = sum(1 for p in posts if p.get("category") == "thought_leadership")
    tl_share = round(tl_count / len(posts) * 100, 1) if posts else 0

    strategic_cards = {
        "alcance": {
            "total_impressions": total_impressions,
            "avg_engagement_no_link": avg_eng_no_link,
        },
        "negocio": {
            "conversation_ratio": conv_ratio,
            "conversion_count": len(conversion_posts),
        },
        "autoridad": {
            "best_topic": best_topic["topic"],
            "best_topic_median_eng": best_topic["median_engagement"],
            "thought_leadership_share": tl_share,
        },
    }

    result = {
        "total_posts": len(posts),
        "total_reposts": len(reposts),
        "total_reactions": sum(p.get("reactions", 0) for p in posts),
        "total_comments": sum(p.get("comments", 0) for p in posts),
        "avg_engagement": round(sum(p.get("engagement", 0) for p in posts) / len(posts), 1),
        "total_impressions": sum(p.get("impressions", 0) for p in posts),
        "posts_by_month": {
            "labels": sorted_months,
            "values": [posts_by_month[m] for m in sorted_months],
            "avg_impressions": [avg_impressions_by_month[m] for m in sorted_months],
        },
        "posts_by_week": {
            "labels": sorted_weeks,
            "values": [posts_by_week[w] for w in sorted_weeks],
        },
        "posts_by_day": {
            "labels": day_names,
            "values": [posts_by_day[i] for i in range(7)],
        },
        "posts_by_hour": {
            "labels": list(range(24)),
            "values": [posts_by_hour[h] for h in range(24)],
        },
        "content_types": {
            "labels": list(content_types.keys()),
            "values": list(content_types.values()),
        },
        "categories": {
            "labels": list(categories.keys()),
            "values": list(categories.values()),
        },
        "sentiments": {
            "labels": list(sentiments.keys()),
            "values": list(sentiments.values()),
        },
        "image_types": {
            "labels": list(image_types.keys()),
            "values": list(image_types.values()),
        },
        "top_keywords": [{"text": kw, "count": c} for kw, c in top_keywords],
        "length_buckets": {
            "labels": list(buckets.keys()),
            "values": list(buckets.values()),
        },
        "top_posts": top_posts_data,
        "all_posts": all_posts_data,
        "length_vs_engagement": scatter_data,
        "funnel_analysis": funnel_analysis,
        "topic_matrix": topic_matrix,
        "day_hour_heatmap": day_hour_heatmap,
        "strategic_cards": strategic_cards,
    }

    if has_engagement:
        result["engagement_evolution"] = {
            "labels": sorted_months,
            "values": [avg_engagement_by_month[m] for m in sorted_months],
            "impressions": [avg_impressions_by_month[m] for m in sorted_months],
        }
        result["posts_by_day"]["avg_engagement"] = [
            round(engagement_by_day[i] / day_counts[i], 1) if day_counts[i] else 0
            for i in range(7)
        ]
        result["posts_by_hour"]["avg_engagement"] = [
            round(engagement_by_hour[h] / hour_counts[h], 1) if hour_counts[h] else 0
            for h in range(24)
        ]
        result["engagement_by_type"] = {
            "labels": list(avg_eng_by_type.keys()),
            "values": list(avg_eng_by_type.values()),
        }
        result["image_engagement"] = {
            "labels": ["With Image", "Without Image"],
            "values": [avg_eng_with, avg_eng_without],
            "counts": [len(with_img), len(without_img)],
        }

    return result
