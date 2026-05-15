"""Statische PDF-Baseline-Matrix für Push-Timing-Empfehlungen.

Quelle: push_timing_analyse.pdf (Stand 13.05.2026, Basis 14.916 Pushes 30.05.2024-13.05.2026).
18 Stunden (06-23) x 7 Wochentage (Python-Konvention: Mo=0..So=6) = 126 Zellen.
"""

from __future__ import annotations
from typing import Optional


def _stars_from_or(avg_or: float) -> int:
    if avg_or >= 6.8:
        return 3
    if avg_or >= 6.0:
        return 2
    if avg_or >= 5.3:
        return 1
    return 0


PDF_TOTAL_PUSHES_ANALYZED = 14916
PDF_OVERALL_AVG = 5.44
PDF_BEST_HOUR = 21
PDF_BEST_OR = 7.21

PDF_KPI = {
    "target_avg_or": 6.0,
    "max_pushes_per_day": 20,
    "mandatory_hours": [20, 21],
    "avoid_hours": [10, 11],
    "min_interval_min": 90,
}

PDF_GOLDEN_RULES = [
    "21+20 Uhr taeglich fixieren — bestes Zeitfenster, immer belegen.",
    "10-11 Uhr stark reduzieren — Totzonen Ø 4,5%, nur fuer Eilmeldungen.",
    "Mo 07-08 und So 08 Uhr nutzen — Top-Slots der Woche (Mo 6,9%, So 7,3%).",
    "Maximal 20 geplante Pushes/Tag — Eilmeldungen immer extra.",
    "Abstand >90 Min zwischen Pushes einhalten.",
]

PDF_HOUR_AVG = {
    6: 6.46, 7: 6.11, 8: 5.87, 9: 4.91, 10: 4.51, 11: 4.53,
    12: 5.01, 13: 5.05, 14: 4.96, 15: 5.56, 16: 5.61, 17: 5.21,
    18: 6.08, 19: 5.99, 20: 6.28, 21: 7.21, 22: 5.67, 23: 5.29,
}

PDF_WEEKDAY_AVG = {
    0: 6.11,  # Mo
    1: 5.38,  # Di
    2: 5.34,  # Mi
    3: 5.38,  # Do
    4: 5.14,  # Fr
    5: 5.28,  # Sa
    6: 5.47,  # So
}

# Format: (hour, weekday) -> {"avg_or", "count", "top_cat", "stars"}
# Weekday: Mo=0, Di=1, Mi=2, Do=3, Fr=4, Sa=5, So=6
# None bei top_cat = "keine Daten" (n < 10).
_M: dict = {
    # 06:00 — Frueh-Slot, oft Eilmeldungen
    (6, 0): None,
    (6, 1): {"avg_or": 4.95, "count": 12, "top_cat": "news"},
    (6, 2): {"avg_or": 6.76, "count": 16, "top_cat": "news"},
    (6, 3): {"avg_or": 6.86, "count": 21, "top_cat": "politik"},
    (6, 4): None,
    (6, 5): {"avg_or": 6.67, "count": 21, "top_cat": "news"},
    (6, 6): None,
    # 07:00 — Morgen-Fenster
    (7, 0): {"avg_or": 6.48, "count": 19, "top_cat": "news"},
    (7, 1): {"avg_or": 6.22, "count": 83, "top_cat": "news"},
    (7, 2): {"avg_or": 5.59, "count": 80, "top_cat": "geld"},
    (7, 3): {"avg_or": 6.24, "count": 88, "top_cat": "news"},
    (7, 4): {"avg_or": 6.14, "count": 81, "top_cat": "news"},
    (7, 5): {"avg_or": 6.11, "count": 75, "top_cat": "news"},
    (7, 6): {"avg_or": 6.75, "count": 20, "top_cat": "news"},
    # 08:00
    (8, 0): {"avg_or": 6.92, "count": 95, "top_cat": "news"},
    (8, 1): {"avg_or": 5.53, "count": 99, "top_cat": "news"},
    (8, 2): {"avg_or": 5.61, "count": 87, "top_cat": "news"},
    (8, 3): {"avg_or": 5.54, "count": 110, "top_cat": "news"},
    (8, 4): {"avg_or": 5.40, "count": 126, "top_cat": "news"},
    (8, 5): {"avg_or": 5.39, "count": 119, "top_cat": "news"},
    (8, 6): {"avg_or": 7.32, "count": 72, "top_cat": "regional"},
    # 09:00
    (9, 0): {"avg_or": 6.45, "count": 114, "top_cat": "regional"},
    (9, 1): {"avg_or": 4.59, "count": 153, "top_cat": "news"},
    (9, 2): {"avg_or": 4.63, "count": 152, "top_cat": "geld"},
    (9, 3): {"avg_or": 4.65, "count": 129, "top_cat": "news"},
    (9, 4): {"avg_or": 4.29, "count": 138, "top_cat": "news"},
    (9, 5): {"avg_or": 4.79, "count": 115, "top_cat": "news"},
    (9, 6): {"avg_or": 5.36, "count": 103, "top_cat": "regional"},
    # 10:00 — Totzone Start
    (10, 0): {"avg_or": 5.72, "count": 126, "top_cat": "news"},
    (10, 1): {"avg_or": 4.39, "count": 152, "top_cat": "regional"},
    (10, 2): {"avg_or": 4.34, "count": 166, "top_cat": "news"},
    (10, 3): {"avg_or": 4.37, "count": 157, "top_cat": "news"},
    (10, 4): {"avg_or": 3.89, "count": 179, "top_cat": "news"},
    (10, 5): {"avg_or": 4.14, "count": 180, "top_cat": "regional"},
    (10, 6): {"avg_or": 5.19, "count": 131, "top_cat": "news"},
    # 11:00 — Totzone
    (11, 0): {"avg_or": 5.51, "count": 132, "top_cat": "news"},
    (11, 1): {"avg_or": 4.19, "count": 183, "top_cat": "regional"},
    (11, 2): {"avg_or": 4.40, "count": 168, "top_cat": "news"},
    (11, 3): {"avg_or": 4.58, "count": 163, "top_cat": "news"},
    (11, 4): {"avg_or": 4.25, "count": 199, "top_cat": "news"},
    (11, 5): {"avg_or": 4.27, "count": 177, "top_cat": "news"},
    (11, 6): {"avg_or": 4.93, "count": 133, "top_cat": "regional"},
    # 12:00
    (12, 0): {"avg_or": 5.90, "count": 149, "top_cat": "regional"},
    (12, 1): {"avg_or": 5.07, "count": 176, "top_cat": "news"},
    (12, 2): {"avg_or": 5.05, "count": 158, "top_cat": "news"},
    (12, 3): {"avg_or": 4.99, "count": 161, "top_cat": "news"},
    (12, 4): {"avg_or": 4.49, "count": 174, "top_cat": "news"},
    (12, 5): {"avg_or": 4.62, "count": 180, "top_cat": "news"},
    (12, 6): {"avg_or": 5.11, "count": 138, "top_cat": "news"},
    # 13:00
    (13, 0): {"avg_or": 5.63, "count": 118, "top_cat": "news"},
    (13, 1): {"avg_or": 4.70, "count": 135, "top_cat": "news"},
    (13, 2): {"avg_or": 4.87, "count": 146, "top_cat": "news"},
    (13, 3): {"avg_or": 5.23, "count": 133, "top_cat": "news"},
    (13, 4): {"avg_or": 4.53, "count": 159, "top_cat": "regional"},
    (13, 5): {"avg_or": 5.11, "count": 117, "top_cat": "news"},
    (13, 6): {"avg_or": 5.51, "count": 133, "top_cat": "news"},
    # 14:00
    (14, 0): {"avg_or": 5.66, "count": 147, "top_cat": "news"},
    (14, 1): {"avg_or": 4.97, "count": 152, "top_cat": "news"},
    (14, 2): {"avg_or": 4.99, "count": 127, "top_cat": "news"},
    (14, 3): {"avg_or": 4.97, "count": 156, "top_cat": "news"},
    (14, 4): {"avg_or": 4.57, "count": 161, "top_cat": "regional"},
    (14, 5): {"avg_or": 4.76, "count": 144, "top_cat": "news"},
    (14, 6): {"avg_or": 4.86, "count": 138, "top_cat": "news"},
    # 15:00
    (15, 0): {"avg_or": 5.64, "count": 145, "top_cat": "news"},
    (15, 1): {"avg_or": 5.40, "count": 111, "top_cat": "news"},
    (15, 2): {"avg_or": 5.63, "count": 122, "top_cat": "unterhaltung"},
    (15, 3): {"avg_or": 5.50, "count": 103, "top_cat": "news"},
    (15, 4): {"avg_or": 5.78, "count": 111, "top_cat": "news"},
    (15, 5): {"avg_or": 5.41, "count": 118, "top_cat": "news"},
    (15, 6): {"avg_or": 5.56, "count": 128, "top_cat": "news"},
    # 16:00
    (16, 0): {"avg_or": 5.91, "count": 129, "top_cat": "news"},
    (16, 1): {"avg_or": 5.30, "count": 126, "top_cat": "unterhaltung"},
    (16, 2): {"avg_or": 5.47, "count": 122, "top_cat": "news"},
    (16, 3): {"avg_or": 5.28, "count": 132, "top_cat": "unterhaltung"},
    (16, 4): {"avg_or": 5.63, "count": 104, "top_cat": "unterhaltung"},
    (16, 5): {"avg_or": 6.14, "count": 112, "top_cat": "news"},
    (16, 6): {"avg_or": 5.60, "count": 117, "top_cat": "news"},
    # 17:00
    (17, 0): {"avg_or": 5.67, "count": 163, "top_cat": "news"},
    (17, 1): {"avg_or": 5.57, "count": 136, "top_cat": "news"},
    (17, 2): {"avg_or": 5.85, "count": 114, "top_cat": "news"},
    (17, 3): {"avg_or": 5.48, "count": 135, "top_cat": "regional"},
    (17, 4): {"avg_or": 4.91, "count": 156, "top_cat": "unterhaltung"},
    (17, 5): {"avg_or": 5.16, "count": 141, "top_cat": "news"},
    (17, 6): {"avg_or": 4.46, "count": 245, "top_cat": "news"},
    # 18:00 — Abend-Start
    (18, 0): {"avg_or": 6.99, "count": 137, "top_cat": "news"},
    (18, 1): {"avg_or": 6.48, "count": 111, "top_cat": "news"},
    (18, 2): {"avg_or": 5.89, "count": 143, "top_cat": "news"},
    (18, 3): {"avg_or": 6.08, "count": 119, "top_cat": "regional"},
    (18, 4): {"avg_or": 5.91, "count": 118, "top_cat": "news"},
    (18, 5): {"avg_or": 5.57, "count": 119, "top_cat": "news"},
    (18, 6): {"avg_or": 5.71, "count": 155, "top_cat": "news"},
    # 19:00
    (19, 0): {"avg_or": 6.35, "count": 167, "top_cat": "news"},
    (19, 1): {"avg_or": 6.26, "count": 130, "top_cat": "regional"},
    (19, 2): {"avg_or": 5.87, "count": 134, "top_cat": "news"},
    (19, 3): {"avg_or": 5.49, "count": 131, "top_cat": "news"},
    (19, 4): {"avg_or": 6.46, "count": 128, "top_cat": "news"},
    (19, 5): {"avg_or": 5.66, "count": 145, "top_cat": "regional"},
    (19, 6): {"avg_or": 5.81, "count": 147, "top_cat": "news"},
    # 20:00 — Primetime
    (20, 0): {"avg_or": 7.17, "count": 129, "top_cat": "unterhaltung"},
    (20, 1): {"avg_or": 6.33, "count": 113, "top_cat": "unterhaltung"},
    (20, 2): {"avg_or": 6.57, "count": 123, "top_cat": "news"},
    (20, 3): {"avg_or": 6.34, "count": 162, "top_cat": "news"},
    (20, 4): {"avg_or": 6.19, "count": 118, "top_cat": "news"},
    (20, 5): {"avg_or": 6.13, "count": 160, "top_cat": "news"},
    (20, 6): {"avg_or": 5.52, "count": 173, "top_cat": "politik"},
    # 21:00 — Goldene Stunde
    (21, 0): {"avg_or": 7.53, "count": 92, "top_cat": "news"},
    (21, 1): {"avg_or": 7.54, "count": 93, "top_cat": "news"},
    (21, 2): {"avg_or": 7.06, "count": 110, "top_cat": "news"},
    (21, 3): {"avg_or": 7.15, "count": 99, "top_cat": "regional"},
    (21, 4): {"avg_or": 7.44, "count": 84, "top_cat": "news"},
    (21, 5): {"avg_or": 7.11, "count": 96, "top_cat": "news"},
    (21, 6): {"avg_or": 6.73, "count": 101, "top_cat": "news"},
    # 22:00
    (22, 0): {"avg_or": 6.06, "count": 99, "top_cat": "news"},
    (22, 1): {"avg_or": 6.20, "count": 75, "top_cat": "news"},
    (22, 2): {"avg_or": 5.05, "count": 110, "top_cat": "news"},
    (22, 3): {"avg_or": 5.48, "count": 121, "top_cat": "news"},
    (22, 4): {"avg_or": 5.39, "count": 94, "top_cat": "news"},
    (22, 5): {"avg_or": 5.64, "count": 142, "top_cat": "news"},
    (22, 6): {"avg_or": 6.01, "count": 119, "top_cat": "regional"},
    # 23:00
    (23, 0): {"avg_or": 5.35, "count": 29, "top_cat": "news"},
    (23, 1): {"avg_or": 4.91, "count": 15, "top_cat": "news"},
    (23, 2): {"avg_or": 5.11, "count": 37, "top_cat": "news"},
    (23, 3): {"avg_or": 4.67, "count": 39, "top_cat": "news"},
    (23, 4): {"avg_or": 4.89, "count": 24, "top_cat": "news"},
    (23, 5): {"avg_or": 5.67, "count": 39, "top_cat": "news"},
    (23, 6): {"avg_or": 6.30, "count": 30, "top_cat": "news"},
}

# Stars aus avg_or ableiten (1x am Modul-Load)
PDF_OR_MATRIX: dict = {}
for _k, _v in _M.items():
    if _v is None:
        PDF_OR_MATRIX[_k] = None
    else:
        PDF_OR_MATRIX[_k] = dict(_v)
        PDF_OR_MATRIX[_k]["stars"] = _stars_from_or(_v["avg_or"])


def baseline_for(hour: int, weekday: int) -> Optional[dict]:
    """Liefert PDF-Baseline fuer (hour, weekday=Python-Konvention Mo=0..So=6).

    Returns None bei: Stunde ausserhalb 6-23, oder Zelle mit zu wenig Daten (n<10).
    """
    if hour < 6 or hour > 23:
        return None
    if weekday < 0 or weekday > 6:
        return None
    return PDF_OR_MATRIX.get((hour, weekday))


def blend_with_db(hour: int, weekday: int, db_avg_or: Optional[float], db_count: int) -> dict:
    """Gewichteter Blend von PDF-Baseline und DB-Live-Daten.

    db_count < 10: PDF gewinnt komplett.
    db_count >= 10: linearer Blend, beide gewichtet nach Count.
    PDF-Zelle fehlt: DB-Daten allein, wenn vorhanden — sonst Stunden-Average.
    """
    pdf = baseline_for(hour, weekday)

    if not pdf:
        if db_avg_or is not None and db_count > 0:
            return {"avg_or": db_avg_or, "count": db_count, "source": "db", "top_cat": None,
                    "stars": _stars_from_or(db_avg_or)}
        hour_avg = PDF_HOUR_AVG.get(hour, PDF_OVERALL_AVG)
        return {"avg_or": hour_avg, "count": 0, "source": "fallback", "top_cat": None,
                "stars": _stars_from_or(hour_avg)}

    if db_avg_or is None or db_count < 10:
        return {**pdf, "source": "pdf"}

    pdf_count = pdf["count"]
    total_w = pdf_count + db_count
    blended_or = (pdf["avg_or"] * pdf_count + db_avg_or * db_count) / total_w
    return {
        "avg_or": round(blended_or, 2),
        "count": db_count,
        "top_cat": pdf["top_cat"],
        "stars": _stars_from_or(blended_or),
        "source": "blend",
    }


def kpi_status(pushed_hours_today: list, current_avg_or: Optional[float],
               n_pushed_today: int, current_hour: int) -> dict:
    """Berechnet KPI-Status gegen PDF-Zielwerte.

    pushed_hours_today: Liste der Stunden, in denen heute schon gepusht wurde.
    current_avg_or: Ø OR der heutigen Pushes (in Prozent).
    n_pushed_today: Anzahl Pushes heute.
    current_hour: aktuelle Stunde — fuer Status der Pflicht-Stunden (offen vs verpasst).
    """
    pushed_set = set(pushed_hours_today)
    mandatory = PDF_KPI["mandatory_hours"]
    avoid = PDF_KPI["avoid_hours"]

    mandatory_filled = [h for h in mandatory if h in pushed_set]
    mandatory_missed = [h for h in mandatory if h not in pushed_set and h < current_hour]
    mandatory_pending = [h for h in mandatory if h not in pushed_set and h >= current_hour]

    avoid_violated = [h for h in avoid if h in pushed_set]

    avg_or_target = PDF_KPI["target_avg_or"]
    avg_or_ok = (current_avg_or is not None and current_avg_or >= avg_or_target)
    pushes_limit = PDF_KPI["max_pushes_per_day"]
    pushes_ok = n_pushed_today <= pushes_limit

    return {
        "target_avg_or": avg_or_target,
        "current_avg_or": current_avg_or,
        "avg_or_ok": avg_or_ok,
        "pushes_today": n_pushed_today,
        "pushes_limit": pushes_limit,
        "pushes_ok": pushes_ok,
        "mandatory_hours": mandatory,
        "mandatory_filled": mandatory_filled,
        "mandatory_missed": mandatory_missed,
        "mandatory_pending": mandatory_pending,
        "mandatory_ok": len(mandatory_missed) == 0,
        "avoid_hours": avoid,
        "avoid_violated": avoid_violated,
        "avoid_ok": len(avoid_violated) == 0,
    }
