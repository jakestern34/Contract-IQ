from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# MLB Contract Intelligence — iPad-friendly flat deployment build
# Everything the app needs lives at the repository root. No folders required.
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DEFAULT_START_YEAR = 2008
DEFAULT_END_YEAR = 2026
DEFAULT_SIMULATIONS = 5000
DEFAULT_DOLLARS_PER_WAR = 9.5

POSITION_GROUPS = {
    "C": "C", "1B": "Corner IF", "2B": "Middle IF", "3B": "Corner IF",
    "SS": "Middle IF", "LF": "OF", "CF": "OF", "RF": "OF", "DH": "DH",
    "SP": "SP", "RP": "RP", "P": "P",
}

HITTER_METRICS = [
    "war", "wrc_plus", "ops_plus", "obp", "slg", "iso", "hr", "bb_pct",
    "k_pct", "baserunning_runs", "defensive_runs", "avg_ev", "barrel_pct",
    "hard_hit_pct", "launch_angle", "bat_speed", "chase_pct", "whiff_pct",
    "sprint_speed"
]
PITCHER_METRICS = [
    "war", "era", "era_plus", "fip", "xfip", "innings", "k_pct", "bb_pct",
    "hr9", "gb_pct", "avg_velocity", "whiff_pct", "chase_pct", "stuff_plus",
    "location_plus", "pitching_plus", "hard_hit_pct_allowed"
]
HITTER_SCOUTING = ["hit_grade", "power_grade", "run_grade", "arm_grade", "field_grade", "overall_grade"]
PITCHER_SCOUTING = ["fastball_grade", "slider_grade", "curveball_grade", "changeup_grade", "control_grade", "overall_grade"]

TEMPLATE_COLUMNS = {
    "contracts_template.csv": "contract_id,player_id,player_name,signing_date,signing_year,signing_age,age,role,position,negotiation_type,team,years,aav_m,guarantee_m,service_years,il_days_last_3y,ped_suspensions,other_suspensions,war,wrc_plus,ops_plus,obp,slg,iso,bb_pct,k_pct,defensive_runs,baserunning_runs,avg_ev,barrel_pct,hard_hit_pct,chase_pct,whiff_pct,era,fip,xfip,innings,avg_velocity,stuff_plus,location_plus,overall_grade,source,source_url".split(','),
    "player_seasons_template.csv": "player_id,player_name,season,age,role,position,war,wrc_plus,ops_plus,obp,slg,iso,hr,bb_pct,k_pct,baserunning_runs,defensive_runs,avg_ev,barrel_pct,hard_hit_pct,launch_angle,bat_speed,chase_pct,whiff_pct,sprint_speed,era,era_plus,fip,xfip,innings,hr9,gb_pct,avg_velocity,stuff_plus,location_plus,pitching_plus,hard_hit_pct_allowed".split(','),
    "scouting_template.csv": "player_id,player_name,scout_year,role,hit_grade,power_grade,run_grade,arm_grade,field_grade,fastball_grade,slider_grade,curveball_grade,changeup_grade,control_grade,overall_grade,source,source_url".split(','),
    "suspensions_template.csv": "player_id,player_name,year,games,category,reason,source,source_url".split(','),
}


# ===== UTILS =====
def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def money_m(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"${value:,.1f}M"


def pct(value: float, decimals: int = 0) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{decimals}f}%"


def weighted_mean(values, weights, default=np.nan):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return default
    return float(np.average(v[mask], weights=w[mask]))


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# ===== MARKET =====
def _percentile_rank(series: pd.Series, value: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna().sort_values().values
    if len(s) == 0:
        return 0.5
    return float(np.searchsorted(s, value, side="right") / len(s))


def _quantile(series: pd.Series, q: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(s.quantile(np.clip(q, 0.01, 0.99)))


def normalize_contract_market(contracts: pd.DataFrame, target_year: int | None = None) -> pd.DataFrame:
    if contracts.empty:
        return contracts.copy()
    df = contracts.copy()
    df["signing_year"] = pd.to_numeric(df["signing_year"], errors="coerce")
    if target_year is None:
        target_year = int(df["signing_year"].dropna().max())
    recent = df[df["signing_year"].between(target_year - 2, target_year)]
    if len(recent) < 12:
        recent = df[df["signing_year"] == target_year]
    if len(recent) < 8:
        recent = df.sort_values("signing_year").tail(min(50, len(df)))

    norm_aav, norm_total, pct_aav = [], [], []
    for _, row in df.iterrows():
        yr = row.get("signing_year")
        peer = df[df["signing_year"] == yr]
        if row.get("negotiation_type") in ("Free Agent", "Extension", "Arbitration"):
            same_type = peer[peer.get("negotiation_type") == row.get("negotiation_type")]
            if len(same_type) >= 8:
                peer = same_type
        qa = _percentile_rank(peer.get("aav_m", pd.Series(dtype=float)), float(row.get("aav_m", np.nan)))
        qt = _percentile_rank(peer.get("guarantee_m", pd.Series(dtype=float)), float(row.get("guarantee_m", np.nan)))
        na = _quantile(recent.get("aav_m", pd.Series(dtype=float)), qa)
        nt = _quantile(recent.get("guarantee_m", pd.Series(dtype=float)), qt)
        if not np.isfinite(na):
            na = float(row.get("aav_m", np.nan))
        if not np.isfinite(nt):
            nt = float(row.get("guarantee_m", np.nan))
        norm_aav.append(na)
        norm_total.append(nt)
        pct_aav.append(qa)
    df["market_percentile"] = pct_aav
    df["normalized_aav_m"] = norm_aav
    df["normalized_guarantee_m"] = norm_total
    return df


# ===== SIMILARITY =====
DEFAULT_WEIGHTS = {
    "age": 1.4,
    "war": 2.2,
    "wrc_plus": 1.3,
    "ops_plus": 1.0,
    "obp": 0.6,
    "slg": 0.6,
    "iso": 0.6,
    "bb_pct": 0.6,
    "k_pct": 0.6,
    "defensive_runs": 0.7,
    "baserunning_runs": 0.4,
    "avg_ev": 0.7,
    "barrel_pct": 0.7,
    "hard_hit_pct": 0.7,
    "chase_pct": 0.5,
    "whiff_pct": 0.5,
    "era": 1.0,
    "fip": 1.3,
    "xfip": 1.0,
    "innings": 0.9,
    "avg_velocity": 0.8,
    "stuff_plus": 0.8,
    "location_plus": 0.7,
}


def _position_bonus(query_position: str, comp_position: str) -> float:
    if not query_position or not comp_position:
        return 0.0
    if query_position == comp_position:
        return 0.10
    if POSITION_GROUPS.get(query_position) == POSITION_GROUPS.get(comp_position):
        return 0.05
    return -0.05


def similarity_scores(query: dict, contracts: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    if contracts.empty:
        return contracts.copy()
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    df = contracts.copy()
    distances = []
    coverage = []
    for _, row in df.iterrows():
        num = 0.0
        den = 0.0
        used = 0
        for metric, w in weights.items():
            qv = query.get(metric)
            rv = row.get(metric)
            try:
                qv = float(qv)
                rv = float(rv)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(qv) or not np.isfinite(rv):
                continue
            # Scale by cross-sectional dispersion when available; robust fallback.
            col = pd.to_numeric(df.get(metric), errors="coerce") if metric in df.columns else pd.Series(dtype=float)
            scale = float(col.std()) if len(col) else 1.0
            if not np.isfinite(scale) or scale < 1e-6:
                scale = max(abs(qv) * 0.15, 1.0)
            num += w * ((qv - rv) / scale) ** 2
            den += w
            used += 1
        dist = np.sqrt(num / den) if den else 4.0
        score = 100.0 * np.exp(-0.42 * dist)
        score += 100 * _position_bonus(str(query.get("position", "")), str(row.get("position", "")))
        if query.get("negotiation_type") and row.get("negotiation_type"):
            score += 5 if query["negotiation_type"] == row["negotiation_type"] else -8
        distances.append(float(np.clip(score, 0, 100)))
        coverage.append(used / max(len(weights), 1))
    df["similarity_score"] = distances
    df["metric_coverage"] = coverage
    df["effective_score"] = df["similarity_score"] * (0.75 + 0.25 * df["metric_coverage"])
    return df.sort_values(["effective_score", "signing_year"], ascending=[False, False]).reset_index(drop=True)


# ===== PROJECTION =====
def scouting_weight(age: float, service_years: float) -> float:
    # Scouting matters most before MLB performance has fully stabilized.
    age_factor = np.clip((29 - float(age)) / 8, 0, 1)
    service_factor = np.clip((4.5 - float(service_years)) / 4.5, 0, 1)
    return float(0.45 * age_factor * service_factor)


def generic_aging_delta(age: float, role: str) -> float:
    age = float(age)
    if age <= 23:
        return 0.20
    if age <= 25:
        return 0.12
    if age <= 27:
        return 0.05
    if age <= 29:
        return -0.08
    if age <= 31:
        return -0.22 if role == "Hitter" else -0.27
    if age <= 33:
        return -0.35 if role == "Hitter" else -0.42
    return -0.50 if role == "Hitter" else -0.58


def _build_transitions(seasons: pd.DataFrame) -> pd.DataFrame:
    if seasons.empty or "player_id" not in seasons.columns:
        return pd.DataFrame()
    df = seasons.copy().sort_values(["player_id", "season"])
    nxt = df[["player_id", "season", "war"]].copy()
    nxt["season"] = nxt["season"] - 1
    nxt = nxt.rename(columns={"war": "next_war"})
    merged = df.merge(nxt, on=["player_id", "season"], how="left")
    merged["war_delta"] = pd.to_numeric(merged["next_war"], errors="coerce") - pd.to_numeric(merged["war"], errors="coerce")
    return merged.dropna(subset=["war_delta"])


def select_cohort(query: dict, seasons: pd.DataFrame, n: int = 80) -> pd.DataFrame:
    if seasons.empty:
        return pd.DataFrame()
    df = seasons.copy()
    if "role" in df.columns:
        df = df[df["role"].astype(str) == str(query.get("role", "Hitter"))]
    qage = float(query.get("age", 27))
    df = df[pd.to_numeric(df.get("age"), errors="coerce").between(qage - 2.5, qage + 2.5)]
    qpos = str(query.get("position", ""))
    if qpos and "position" in df.columns:
        qgroup = POSITION_GROUPS.get(qpos, qpos)
        same = df[df["position"].map(lambda x: POSITION_GROUPS.get(str(x), str(x))) == qgroup]
        if len(same) >= 25:
            df = same
    if df.empty:
        return df
    features = ["war", "wrc_plus", "ops_plus"] if query.get("role") == "Hitter" else ["war", "fip", "k_pct", "bb_pct"]
    d = np.zeros(len(df))
    used = np.zeros(len(df))
    for feat in features:
        if feat not in df.columns or query.get(feat) is None:
            continue
        vals = pd.to_numeric(df[feat], errors="coerce")
        q = float(query[feat])
        scale = vals.std()
        if not np.isfinite(scale) or scale < 1e-6:
            scale = max(abs(q) * 0.15, 1.0)
        ok = vals.notna().values
        d[ok] += ((vals[ok].values - q) / scale) ** 2
        used[ok] += 1
    df = df.copy()
    df["cohort_distance"] = np.sqrt(d / np.maximum(used, 1))
    return df.sort_values("cohort_distance").head(n)


def project_player(query: dict, seasons: pd.DataFrame, years: int = 8, simulations: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    current_war = float(query.get("war", 2.5) or 2.5)
    age0 = float(query.get("age", 27))
    role = str(query.get("role", "Hitter"))
    service = float(query.get("service_years", 3.0) or 3.0)
    overall = float(query.get("overall_grade", 50) or 50)
    scout_w = scouting_weight(age0, service)
    scouting_adj = scout_w * ((overall - 50) / 10) * 0.22

    transitions = _build_transitions(seasons)
    cohort = select_cohort(query, transitions if not transitions.empty else seasons, n=100)
    deltas = pd.to_numeric(cohort.get("war_delta"), errors="coerce").dropna().values if not cohort.empty and "war_delta" in cohort.columns else np.array([])

    injury_days = float(query.get("il_days_last_3y", 0) or 0)
    durability_penalty = np.clip(injury_days / 360.0, 0, 0.25)

    paths = np.zeros((simulations, years))
    prev = np.full(simulations, current_war)
    for y in range(years):
        age = age0 + y
        generic = generic_aging_delta(age, role)
        if len(deltas) >= 15:
            sampled = rng.choice(deltas, size=simulations, replace=True)
            delta = 0.58 * sampled + 0.42 * generic
        else:
            delta = rng.normal(generic, 0.55, size=simulations)
        if y <= 2:
            delta = delta + scouting_adj * (1 - y / 3)
        noise = rng.normal(0, 0.45 + 0.05 * y, size=simulations)
        availability = np.clip(rng.normal(1 - durability_penalty, 0.07 + 0.01 * y, size=simulations), 0.45, 1.05)
        nxt = np.maximum(-0.5, (prev + delta + noise) * availability)
        paths[:, y] = nxt
        prev = nxt

    ages = np.arange(int(age0), int(age0) + years)
    summary = pd.DataFrame({
        "age": ages,
        "p10_war": np.quantile(paths, 0.10, axis=0),
        "p25_war": np.quantile(paths, 0.25, axis=0),
        "median_war": np.quantile(paths, 0.50, axis=0),
        "p75_war": np.quantile(paths, 0.75, axis=0),
        "p90_war": np.quantile(paths, 0.90, axis=0),
    })
    return summary, paths, cohort


# ===== MLB_API =====
BASE = "https://statsapi.mlb.com/api/v1"
TIMEOUT = 20


def search_player(name: str) -> list[dict]:
    if not name.strip():
        return []
    r = requests.get(f"{BASE}/people/search", params={"names": name.strip(), "sportIds": 1}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json().get("people", [])


def get_player(player_id: int) -> dict:
    r = requests.get(f"{BASE}/people/{player_id}", params={"hydrate": "currentTeam"}, timeout=TIMEOUT)
    r.raise_for_status()
    people = r.json().get("people", [])
    return people[0] if people else {}


def get_season_stats(player_id: int, group: str, season: int | None = None) -> dict:
    season = season or date.today().year
    params = {"stats": "season", "group": group, "season": season}
    r = requests.get(f"{BASE}/people/{player_id}/stats", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    stats = r.json().get("stats", [])
    splits = stats[0].get("splits", []) if stats else []
    return splits[0].get("stat", {}) if splits else {}


def player_prefill(name: str, season: int | None = None) -> dict:
    matches = search_player(name)
    if not matches:
        return {}
    p = matches[0]
    player_id = p.get("id")
    full = get_player(player_id)
    primary_position = (full.get("primaryPosition") or {}).get("abbreviation", "")
    group = "pitching" if primary_position == "P" else "hitting"
    stats = get_season_stats(player_id, group, season)
    return {
        "player_id": player_id,
        "name": full.get("fullName", name),
        "birth_date": full.get("birthDate"),
        "age": full.get("currentAge"),
        "height": full.get("height"),
        "weight": full.get("weight"),
        "bats": (full.get("batSide") or {}).get("code"),
        "throws": (full.get("pitchHand") or {}).get("code"),
        "position": primary_position,
        "team": (full.get("currentTeam") or {}).get("name"),
        "stats": stats,
        "group": group,
    }


# ===== VALUATION =====
def risk_multiplier(query: dict) -> tuple[float, list[str]]:
    notes = []
    il_days = float(query.get("il_days_last_3y", 0) or 0)
    injury_discount = min(0.12, (il_days / 365.0) * 0.09)
    if injury_discount > 0.01:
        notes.append(f"Durability adjustment: -{injury_discount*100:.1f}%")

    ped = int(query.get("ped_suspensions", 0) or 0)
    other = int(query.get("other_suspensions", 0) or 0)
    suspension_discount = min(0.09, ped * 0.025 + other * 0.01)
    if suspension_discount > 0:
        notes.append(f"Verified suspension-risk adjustment: -{suspension_discount*100:.1f}%")

    manual = float(query.get("manual_risk_adjustment_pct", 0) or 0) / 100.0
    if abs(manual) > 1e-9:
        notes.append(f"Analyst qualitative adjustment: {manual*100:+.1f}%")
    mult = (1 - injury_discount) * (1 - suspension_discount) * (1 + manual)
    return float(np.clip(mult, 0.72, 1.15)), notes


def leverage_multiplier(query: dict) -> tuple[float, str]:
    # Input is -10 to +10; intentionally modest to keep market leverage from overpowering fundamentals.
    leverage = float(query.get("leverage_score", 0) or 0)
    mult = 1 + np.clip(leverage, -10, 10) * 0.007
    return float(mult), f"Market leverage: {leverage:+.0f}/10 ({(mult-1)*100:+.1f}%)"


def value_contract(query: dict, contracts: pd.DataFrame, projection: pd.DataFrame,
                   target_year: int, dollars_per_war: float = 9.5, top_n: int = 15) -> dict:
    normalized = normalize_contract_market(contracts, target_year)
    comps = similarity_scores(query, normalized)
    comps = comps.head(max(top_n, 5)).copy()
    if comps.empty:
        return {}

    weights = np.maximum(pd.to_numeric(comps["effective_score"], errors="coerce").fillna(0).values, 1) ** 2
    comp_aav = weighted_mean(comps["normalized_aav_m"], weights)
    comp_total = weighted_mean(comps["normalized_guarantee_m"], weights)
    comp_years = weighted_mean(comps["years"], weights, default=4.0)

    med = projection["median_war"].values if not projection.empty else np.array([float(query.get("war", 2.5))])
    projected_good_years = int(np.clip(np.sum(med >= 1.3), 1, 12))
    age = float(query.get("age", 28))
    age_cap = int(np.clip(40 - age, 1, 12))
    fundamental_years = min(projected_good_years, age_cap)
    years = int(np.clip(round(0.62 * comp_years + 0.38 * fundamental_years), 1, 12))

    yearly_war = med[:years]
    if len(yearly_war) < years:
        yearly_war = np.pad(yearly_war, (0, years-len(yearly_war)), constant_values=max(med[-1] - 0.4, 0.3))
    # Fundamental value uses market $/WAR with a soft floor for roster/option value and discounts distant seasons.
    undiscounted = np.maximum(yearly_war, 0.35) * dollars_per_war
    pv_fundamental = float(np.sum(undiscounted / ((1.06) ** np.arange(years))))
    fundamental_aav = pv_fundamental / years

    # Blend transparent precedent and fundamental engines.
    base_total = 0.64 * comp_total + 0.36 * pv_fundamental
    base_aav = 0.64 * comp_aav + 0.36 * fundamental_aav

    r_mult, risk_notes = risk_multiplier(query)
    l_mult, leverage_note = leverage_multiplier(query)
    expected_total = base_total * r_mult * l_mult
    expected_aav = base_aav * r_mult * l_mult

    # Keep total/AAV internally coherent while preserving comp-derived signal.
    coherent_total = 0.65 * expected_total + 0.35 * (expected_aav * years)
    expected_total = coherent_total
    expected_aav = expected_total / years

    avg_sim = weighted_mean(comps["similarity_score"], weights, default=50)
    coverage = weighted_mean(comps["metric_coverage"], weights, default=0.4)
    confidence = float(np.clip(0.52 * avg_sim + 48 * coverage, 35, 95))
    spread = np.clip(0.22 - confidence / 700, 0.085, 0.18)

    return {
        "years": years,
        "aav_m": expected_aav,
        "guarantee_m": expected_total,
        "low_m": expected_total * (1 - spread),
        "high_m": expected_total * (1 + spread),
        "team_case_m": expected_total * (0.90 + 0.02 * coverage),
        "agent_case_m": expected_total * (1.10 + 0.04 * (1 - coverage)),
        "confidence": confidence,
        "comp_aav_m": comp_aav,
        "comp_total_m": comp_total,
        "fundamental_total_m": pv_fundamental,
        "risk_multiplier": r_mult,
        "leverage_multiplier": l_mult,
        "notes": risk_notes + [leverage_note],
        "comps": comps,
    }


def term_optimizer(projection: pd.DataFrame, dollars_per_war: float, market_aav_anchor: float, max_years: int = 10) -> pd.DataFrame:
    rows = []
    med = projection["median_war"].values
    for years in range(1, max_years + 1):
        w = med[:years]
        if len(w) < years:
            w = np.pad(w, (0, years-len(w)), constant_values=max(med[-1] - 0.4, 0.25))
        fundamental = float(np.sum(np.maximum(w, 0.35) * dollars_per_war / (1.06 ** np.arange(years))))
        implied_aav = 0.55 * (fundamental / years) + 0.45 * market_aav_anchor
        guarantee = implied_aav * years
        rows.append({"years": years, "implied_aav_m": implied_aav, "guarantee_m": guarantee, "projected_war": float(np.sum(w))})
    return pd.DataFrame(rows)


# ===== FLAT DATA STORE =====
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _load_flat(kind: str, use_demo: bool = True) -> pd.DataFrame:
    frames = []
    if use_demo:
        frames.append(_read_csv(ROOT / f"demo_{kind}.csv"))
    live = _read_csv(ROOT / f"live_{kind}.csv")
    if not live.empty:
        frames.append(live)
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def load_contracts(use_demo: bool = True) -> pd.DataFrame:
    return _load_flat("contracts", use_demo)


def load_player_seasons(use_demo: bool = True) -> pd.DataFrame:
    return _load_flat("player_seasons", use_demo)


def load_scouting(use_demo: bool = True) -> pd.DataFrame:
    return _load_flat("scouting", use_demo)


def load_suspensions(use_demo: bool = True) -> pd.DataFrame:
    return _load_flat("suspensions", use_demo)


def dataset_status() -> dict:
    status = {}
    for name in ["contracts", "player_seasons", "scouting", "suspensions", "mlb_current_players"]:
        path = ROOT / f"live_{name}.csv"
        status[name] = {
            "exists": path.exists(),
            "rows": len(_read_csv(path)) if path.exists() else 0,
            "path": path.name,
        }
    return status


def template_bytes(filename: str) -> bytes:
    cols = TEMPLATE_COLUMNS[filename]
    return pd.DataFrame(columns=cols).to_csv(index=False).encode("utf-8")


# ===== STREAMLIT APP =====
st.set_page_config(page_title="MLB Contract Intelligence", page_icon="⚾", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1500px;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
.small-note {font-size: .85rem; opacity: .78;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 14px 16px; margin-bottom: 10px;}
@media (max-width: 800px) {
  .block-container {padding-left: .75rem; padding-right: .75rem;}
  [data-testid="column"] {min-width: 100% !important;}
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _load_data(use_demo: bool):
    return load_contracts(use_demo), load_player_seasons(use_demo), load_scouting(use_demo), load_suspensions(use_demo)


def _read_upload(uploaded):
    if uploaded is None:
        return pd.DataFrame()
    return pd.read_csv(uploaded)


def _merge(base: pd.DataFrame, extra: pd.DataFrame):
    if extra is None or extra.empty:
        return base
    if base is None or base.empty:
        return extra.copy()
    return pd.concat([base, extra], ignore_index=True, sort=False)


def _age_from_prefill(prefill):
    if not prefill:
        return 27
    return int(prefill.get("age") or 27)


def _metric_delta_value(stats, key, fallback=0.0):
    try:
        return float(stats.get(key, fallback))
    except Exception:
        return fallback


def _hitter_basic_prefill(stats: dict):
    pa = _metric_delta_value(stats, "plateAppearances", 0)
    bb = _metric_delta_value(stats, "baseOnBalls", 0)
    so = _metric_delta_value(stats, "strikeOuts", 0)
    return {
        "obp": _metric_delta_value(stats, "obp", .330),
        "slg": _metric_delta_value(stats, "slg", .430),
        "hr": _metric_delta_value(stats, "homeRuns", 18),
        "bb_pct": 100 * bb / pa if pa else 8.5,
        "k_pct": 100 * so / pa if pa else 22.0,
    }


def _pitcher_basic_prefill(stats: dict):
    ip = str(stats.get("inningsPitched", "0"))
    try:
        innings = float(ip)
    except ValueError:
        innings = 0
    batters = _metric_delta_value(stats, "battersFaced", 0)
    bb = _metric_delta_value(stats, "baseOnBalls", 0)
    so = _metric_delta_value(stats, "strikeOuts", 0)
    return {
        "era": _metric_delta_value(stats, "era", 4.0),
        "innings": innings,
        "k_pct": 100 * so / batters if batters else 24.0,
        "bb_pct": 100 * bb / batters if batters else 8.0,
    }


def _projection_chart(proj: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=proj["age"], y=proj["p90_war"], mode="lines", name="90th %ile"))
    fig.add_trace(go.Scatter(x=proj["age"], y=proj["p75_war"], mode="lines", name="75th %ile"))
    fig.add_trace(go.Scatter(x=proj["age"], y=proj["median_war"], mode="lines+markers", name="Median"))
    fig.add_trace(go.Scatter(x=proj["age"], y=proj["p25_war"], mode="lines", name="25th %ile"))
    fig.add_trace(go.Scatter(x=proj["age"], y=proj["p10_war"], mode="lines", name="10th %ile"))
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=35, b=10), xaxis_title="Age", yaxis_title="Projected WAR", legend_orientation="h")
    return fig


def _term_chart(term_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=term_df["years"], y=term_df["guarantee_m"], name="Guarantee ($M)"))
    fig.add_trace(go.Scatter(x=term_df["years"], y=term_df["implied_aav_m"], mode="lines+markers", name="AAV ($M)", yaxis="y2"))
    fig.update_layout(
        height=390, margin=dict(l=10, r=10, t=35, b=10), xaxis_title="Contract Years",
        yaxis=dict(title="Guarantee ($M)"), yaxis2=dict(title="AAV ($M)", overlaying="y", side="right"), legend_orientation="h")
    return fig


st.sidebar.title("⚾ MLB Contract Intelligence")
page = st.sidebar.radio("Workspace", ["Valuation Lab", "Comparable Contracts", "Data Manager", "Methodology"])
st.sidebar.caption("MVP v0.1 • precedent + projection + negotiation framework")
_live_status = dataset_status()
_have_live_core = _live_status.get("contracts", {}).get("rows", 0) > 0 and _live_status.get("player_seasons", {}).get("rows", 0) > 0
use_demo = st.sidebar.toggle("Use synthetic demo data", value=not _have_live_core, help="Turn this off once validated live_contracts.csv and live_player_seasons.csv files are uploaded to the repository root.")

contracts, seasons, scouting, suspensions = _load_data(use_demo)

# Session uploads persist for the current Streamlit session.
contracts = _merge(contracts, st.session_state.get("upload_contracts", pd.DataFrame()))
seasons = _merge(seasons, st.session_state.get("upload_seasons", pd.DataFrame()))
scouting = _merge(scouting, st.session_state.get("upload_scouting", pd.DataFrame()))
suspensions = _merge(suspensions, st.session_state.get("upload_suspensions", pd.DataFrame()))

if page == "Valuation Lab":
    st.title("MLB Contract Valuation Lab")
    st.caption("Build a negotiation case from historical precedents, cohort-based aging, scouting context, durability and market leverage.")
    if use_demo:
        st.warning("The bundled contract/player dataset is **synthetic demo data** so the model is usable immediately. Replace it with validated/licensed historical data before relying on outputs in a real negotiation.")
    else:
        st.success("Demo records are disabled; valuation is using flat live_*.csv files from the repository root plus any session uploads.")

    with st.expander("Live MLB lookup (optional)", expanded=False):
        c1, c2 = st.columns([3,1])
        lookup = c1.text_input("Player name", placeholder="e.g., Bobby Witt Jr.")
        if c2.button("Fetch MLB data", use_container_width=True):
            try:
                found = player_prefill(lookup, date.today().year)
                if not found:
                    st.error("No MLB player match found.")
                else:
                    st.session_state["prefill"] = found
                    st.success(f"Loaded {found.get('name')} from MLB Stats API.")
            except Exception as e:
                st.error(f"MLB lookup failed: {e}")
        pf = st.session_state.get("prefill", {})
        if pf:
            st.caption(f"Loaded: {pf.get('name')} • {pf.get('team','')} • {pf.get('position','')} • MLBAM {pf.get('player_id','')}")
            st.caption("The official MLB lookup fills identity and basic season stats. Advanced Statcast/FanGraphs fields remain editable and are designed to come from the rolling data-import pipeline.")

    pf = st.session_state.get("prefill", {})
    stats = pf.get("stats", {}) if pf else {}
    live_role = "Pitcher" if pf.get("group") == "pitching" else "Hitter"
    live_basic = _pitcher_basic_prefill(stats) if live_role == "Pitcher" else _hitter_basic_prefill(stats)

    st.subheader("1. Player & negotiation context")
    c1, c2, c3, c4 = st.columns(4)
    player_name = c1.text_input("Player", value=pf.get("name", "Underlying Player"))
    role = c2.selectbox("Role", ["Hitter", "Pitcher"], index=1 if live_role == "Pitcher" else 0)
    negotiation_type = c3.selectbox("Negotiation", ["Free Agent", "Extension", "Arbitration"])
    age = c4.number_input("Age at contract start", 18, 45, _age_from_prefill(pf), 1)

    positions = ["C","1B","2B","3B","SS","LF","CF","RF","DH"] if role == "Hitter" else ["SP","RP"]
    pos_default = pf.get("position")
    if pos_default == "P": pos_default = "SP"
    pos_idx = positions.index(pos_default) if pos_default in positions else 0
    c1, c2, c3, c4 = st.columns(4)
    position = c1.selectbox("Position", positions, index=pos_idx)
    service_years = c2.number_input("MLB service years", 0.0, 20.0, 3.0, .1)
    il_days = c3.number_input("IL days — last 3 years", 0, 1095, 20, 5)
    leverage = c4.slider("Market leverage", -10, 10, 0, help="Scarcity, bidder count, team need, CBT capacity and competitive window. Kept intentionally modest in the model.")

    c1, c2, c3, c4 = st.columns(4)
    ped_susp = c1.number_input("PED suspensions", 0, 5, 0, 1)
    other_susp = c2.number_input("Other verified suspensions", 0, 10, 0, 1)
    manual_risk = c3.slider("Analyst qualitative adjustment", -15, 15, 0, 1, format="%d%%", help="Use only for documented, supportable facts not already captured elsewhere.")
    contract_start = c4.number_input("Target market year", 2008, 2035, date.today().year, 1)

    st.subheader("2. Performance inputs")
    if role == "Hitter":
        c = st.columns(5)
        war = c[0].number_input("WAR", -3.0, 12.0, 4.0, .1)
        wrc_plus = c[1].number_input("wRC+", 40.0, 250.0, 125.0, 1.0)
        ops_plus = c[2].number_input("OPS+", 40.0, 250.0, 125.0, 1.0)
        obp = c[3].number_input("OBP", .200, .550, float(live_basic.get("obp", .350)), .001, format="%.3f")
        slg = c[4].number_input("SLG", .250, .850, float(live_basic.get("slg", .500)), .001, format="%.3f")
        c = st.columns(5)
        iso = c[0].number_input("ISO", .000, .500, .220, .005, format="%.3f")
        hr = c[1].number_input("HR", 0, 80, int(live_basic.get("hr", 25)), 1)
        bb_pct = c[2].number_input("BB %", 0.0, 30.0, float(live_basic.get("bb_pct", 10.0)), .1)
        k_pct = c[3].number_input("K %", 0.0, 50.0, float(live_basic.get("k_pct", 20.0)), .1)
        defensive_runs = c[4].number_input("Defensive value / runs", -30.0, 40.0, 4.0, .5)
        c = st.columns(5)
        baserunning_runs = c[0].number_input("Baserunning runs", -15.0, 20.0, 1.0, .5)
        avg_ev = c[1].number_input("Avg exit velo", 75.0, 105.0, 90.0, .1)
        barrel_pct = c[2].number_input("Barrel %", 0.0, 35.0, 11.0, .1)
        hard_hit_pct = c[3].number_input("Hard-hit %", 10.0, 75.0, 45.0, .1)
        chase_pct = c[4].number_input("Chase %", 10.0, 60.0, 27.0, .1)
        c = st.columns(4)
        whiff_pct = c[0].number_input("Whiff %", 5.0, 60.0, 22.0, .1)
        launch_angle = c[1].number_input("Launch angle", -10.0, 40.0, 13.0, .1)
        bat_speed = c[2].number_input("Bat speed", 55.0, 90.0, 73.0, .1)
        sprint_speed = c[3].number_input("Sprint speed", 20.0, 32.0, 28.0, .1)
    else:
        c = st.columns(5)
        war = c[0].number_input("WAR", -3.0, 12.0, 3.5, .1)
        era = c[1].number_input("ERA", 0.50, 9.00, float(live_basic.get("era", 3.40)), .01)
        fip = c[2].number_input("FIP", 0.50, 9.00, 3.50, .01)
        xfip = c[3].number_input("xFIP", 0.50, 9.00, 3.55, .01)
        innings = c[4].number_input("Innings", 0.0, 260.0, float(live_basic.get("innings", 175.0)), 1.0)
        c = st.columns(5)
        k_pct = c[0].number_input("K %", 0.0, 60.0, float(live_basic.get("k_pct", 28.0)), .1)
        bb_pct = c[1].number_input("BB %", 0.0, 30.0, float(live_basic.get("bb_pct", 7.5)), .1)
        avg_velocity = c[2].number_input("Avg fastball velo", 80.0, 105.0, 95.0, .1)
        whiff_pct = c[3].number_input("Whiff %", 5.0, 60.0, 30.0, .1)
        chase_pct = c[4].number_input("Chase %", 5.0, 60.0, 31.0, .1)
        c = st.columns(5)
        stuff_plus = c[0].number_input("Stuff+", 50.0, 180.0, 110.0, 1.0)
        location_plus = c[1].number_input("Location+", 50.0, 180.0, 102.0, 1.0)
        pitching_plus = c[2].number_input("Pitching+", 50.0, 180.0, 107.0, 1.0)
        gb_pct = c[3].number_input("GB %", 10.0, 80.0, 44.0, .1)
        hr9 = c[4].number_input("HR/9", 0.0, 4.0, 0.95, .01)

    with st.expander("3. Scouting grades (20–80 scale)", expanded=age <= 25 or service_years < 2.5):
        st.caption("Scouting influence automatically decays as MLB track record grows. Current model weight: " + f"{scouting_weight(age, service_years)*100:.0f}%")
        if role == "Hitter":
            c = st.columns(6)
            hit_grade = c[0].slider("Hit", 20, 80, 55, 5)
            power_grade = c[1].slider("Power", 20, 80, 60, 5)
            run_grade = c[2].slider("Run", 20, 80, 50, 5)
            arm_grade = c[3].slider("Arm", 20, 80, 50, 5)
            field_grade = c[4].slider("Field", 20, 80, 50, 5)
            overall_grade = c[5].slider("Overall", 20, 80, 55, 5)
        else:
            c = st.columns(6)
            fastball_grade = c[0].slider("Fastball", 20, 80, 60, 5)
            slider_grade = c[1].slider("Slider", 20, 80, 55, 5)
            curveball_grade = c[2].slider("Curveball", 20, 80, 50, 5)
            changeup_grade = c[3].slider("Changeup", 20, 80, 50, 5)
            control_grade = c[4].slider("Control", 20, 80, 50, 5)
            overall_grade = c[5].slider("Overall", 20, 80, 55, 5)

    with st.expander("4. Model assumptions", expanded=False):
        c1, c2, c3 = st.columns(3)
        projection_years = c1.slider("Projection horizon", 3, 12, 8)
        simulations = c2.select_slider("Monte Carlo paths", options=[1000,2500,5000,10000], value=DEFAULT_SIMULATIONS)
        dollars_per_war = c3.number_input("Current $ / WAR ($M)", 3.0, 20.0, DEFAULT_DOLLARS_PER_WAR, .25)

    query = {
        "player_name": player_name, "role": role, "position": position, "negotiation_type": negotiation_type,
        "age": age, "service_years": service_years, "il_days_last_3y": il_days, "ped_suspensions": ped_susp,
        "other_suspensions": other_susp, "manual_risk_adjustment_pct": manual_risk, "leverage_score": leverage,
        "war": war, "bb_pct": bb_pct, "k_pct": k_pct, "overall_grade": overall_grade,
    }
    if role == "Hitter":
        query.update({"wrc_plus":wrc_plus,"ops_plus":ops_plus,"obp":obp,"slg":slg,"iso":iso,"hr":hr,
                      "defensive_runs":defensive_runs,"baserunning_runs":baserunning_runs,"avg_ev":avg_ev,"barrel_pct":barrel_pct,
                      "hard_hit_pct":hard_hit_pct,"chase_pct":chase_pct,"whiff_pct":whiff_pct,"launch_angle":launch_angle,
                      "bat_speed":bat_speed,"sprint_speed":sprint_speed,"hit_grade":hit_grade,"power_grade":power_grade,
                      "run_grade":run_grade,"arm_grade":arm_grade,"field_grade":field_grade})
    else:
        query.update({"era":era,"fip":fip,"xfip":xfip,"innings":innings,"avg_velocity":avg_velocity,"whiff_pct":whiff_pct,
                      "chase_pct":chase_pct,"stuff_plus":stuff_plus,"location_plus":location_plus,"pitching_plus":pitching_plus,
                      "gb_pct":gb_pct,"hr9":hr9,"fastball_grade":fastball_grade,"slider_grade":slider_grade,
                      "curveball_grade":curveball_grade,"changeup_grade":changeup_grade,"control_grade":control_grade})

    if st.button("Run valuation", type="primary", use_container_width=True):
        with st.spinner("Running cohort projection, precedent matching and contract valuation…"):
            projection, paths, cohort = project_player(query, seasons, years=projection_years, simulations=simulations)
            result = value_contract(query, contracts, projection, target_year=contract_start, dollars_per_war=dollars_per_war)
            st.session_state["last_result"] = result
            st.session_state["last_projection"] = projection
            st.session_state["last_query"] = query
            st.session_state["last_cohort"] = cohort
            st.session_state["last_dpw"] = dollars_per_war

    result = st.session_state.get("last_result")
    projection = st.session_state.get("last_projection")
    last_query = st.session_state.get("last_query")
    if result and projection is not None:
        st.divider()
        st.subheader(f"Valuation output — {last_query.get('player_name','Player')}")
        a,b,c,d,e = st.columns(5)
        a.metric("Implied term", f"{result['years']} years")
        b.metric("Implied AAV", money_m(result['aav_m']))
        c.metric("Implied guarantee", money_m(result['guarantee_m']))
        d.metric("Valuation range", f"{money_m(result['low_m'])} – {money_m(result['high_m'])}")
        e.metric("Model confidence", pct(result['confidence']))

        c1,c2,c3 = st.columns(3)
        c1.metric("Team case", money_m(result['team_case_m']))
        c2.metric("Expected market", money_m(result['guarantee_m']))
        c3.metric("Agent opening case", money_m(result['agent_case_m']))

        st.caption(" • ".join(result.get("notes", [])))

        tab1,tab2,tab3,tab4 = st.tabs(["Precedents", "Upside / aging", "Term optimizer", "Model bridge"])
        with tab1:
            comps = result["comps"].copy()
            show_cols=[c for c in ["player_name","signing_year","signing_age","position","negotiation_type","years","aav_m","guarantee_m","normalized_aav_m","normalized_guarantee_m","similarity_score","metric_coverage","source"] if c in comps.columns]
            st.dataframe(comps[show_cols], use_container_width=True, hide_index=True)
            st.caption("Similarity is era-aware: missing advanced metrics are ignored and remaining features are reweighted. Market-normalized values are percentile-mapped into the target-year/recent contract market.")
        with tab2:
            st.plotly_chart(_projection_chart(projection), use_container_width=True)
            st.dataframe(projection.round(2), use_container_width=True, hide_index=True)
            cohort = st.session_state.get("last_cohort", pd.DataFrame())
            if cohort is not None and not cohort.empty:
                with st.expander("Historical development cohort used"):
                    cols=[c for c in ["player_name","season","age","position","war","war_delta","cohort_distance"] if c in cohort.columns]
                    st.dataframe(cohort[cols].head(30), use_container_width=True, hide_index=True)
        with tab3:
            tdf=term_optimizer(projection, st.session_state.get("last_dpw", DEFAULT_DOLLARS_PER_WAR), result["comp_aav_m"], max_years=min(10,len(projection)+2))
            st.plotly_chart(_term_chart(tdf), use_container_width=True)
            st.dataframe(tdf.round(2), use_container_width=True, hide_index=True)
        with tab4:
            bridge=pd.DataFrame({
                "Engine":["Precedent engine","Fundamental projection engine","Risk / durability multiplier","Market leverage multiplier"],
                "Value":[money_m(result['comp_total_m']), money_m(result['fundamental_total_m']), f"{result['risk_multiplier']:.3f}x", f"{result['leverage_multiplier']:.3f}x"]
            })
            st.dataframe(bridge,use_container_width=True,hide_index=True)
            st.info("Current MVP blends 64% market precedents / 36% fundamental projected value, then applies transparent risk and leverage adjustments. Those weights are intended to be replaced by weights calibrated from historical backtesting once the validated contract-event dataset is populated.")

elif page == "Comparable Contracts":
    st.title("Comparable Contract Explorer")
    st.caption("Explore and normalize the contract-event database." + (" Bundled records are synthetic placeholders until real historical data is imported." if use_demo else " Demo records are disabled."))
    norm = normalize_contract_market(contracts, date.today().year)
    c1,c2,c3,c4 = st.columns(4)
    role_filter=c1.multiselect("Role", sorted(norm["role"].dropna().unique()), default=[])
    type_filter=c2.multiselect("Negotiation", sorted(norm["negotiation_type"].dropna().unique()), default=[])
    year_range=c3.slider("Signing years", int(norm.signing_year.min()), int(norm.signing_year.max()), (max(int(norm.signing_year.min()),2008), int(norm.signing_year.max())))
    min_value=c4.number_input("Min guarantee ($M)",0.0,1000.0,0.0,5.0)
    f=norm[norm.signing_year.between(*year_range) & (norm.guarantee_m>=min_value)]
    if role_filter: f=f[f.role.isin(role_filter)]
    if type_filter: f=f[f.negotiation_type.isin(type_filter)]
    cols=[c for c in ["player_name","signing_year","age","position","negotiation_type","years","aav_m","guarantee_m","market_percentile","normalized_aav_m","normalized_guarantee_m","source"] if c in f.columns]
    st.dataframe(f.sort_values("guarantee_m",ascending=False)[cols], use_container_width=True, hide_index=True)

elif page == "Data Manager":
    st.title("Data Manager")
    st.caption("Designed for rolling updates as seasons progress. CSV uploads are session-scoped here. For persistent data, upload flat live_*.csv files to the repository root; the full automated pipeline can be restored after deployment.")
    status=dataset_status()
    st.subheader("Persistent live datasets")
    rows=[]
    for name,meta in status.items():
        rows.append({"Dataset":name,"Present":meta['exists'],"Rows":meta['rows'],"Repository path":meta['path']})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    st.subheader("Upload / test data in this session")
    c1,c2=st.columns(2)
    with c1:
        up_contracts=st.file_uploader("Contracts / contract-events CSV",type=["csv"],key="upc")
        if up_contracts is not None:
            st.session_state["upload_contracts"]=_read_upload(up_contracts)
            st.success(f"Loaded {len(st.session_state['upload_contracts']):,} contract rows into this session.")
        up_seasons=st.file_uploader("Player seasons CSV",type=["csv"],key="ups")
        if up_seasons is not None:
            st.session_state["upload_seasons"]=_read_upload(up_seasons)
            st.success(f"Loaded {len(st.session_state['upload_seasons']):,} player-season rows into this session.")
    with c2:
        up_scout=st.file_uploader("Scouting grades CSV",type=["csv"],key="upsc")
        if up_scout is not None:
            st.session_state["upload_scouting"]=_read_upload(up_scout)
            st.success(f"Loaded {len(st.session_state['upload_scouting']):,} scouting rows into this session.")
        up_susp=st.file_uploader("Suspensions CSV",type=["csv"],key="upsp")
        if up_susp is not None:
            st.session_state["upload_suspensions"]=_read_upload(up_susp)
            st.success(f"Loaded {len(st.session_state['upload_suspensions']):,} suspension rows into this session.")

    st.subheader("Templates")
    cols=st.columns(4)
    for col,(fname,label) in zip(cols,[
        ("contracts_template.csv","Contract events"),("player_seasons_template.csv","Player seasons"),("scouting_template.csv","Scouting"),("suspensions_template.csv","Suspensions")]):
        col.download_button(label, template_bytes(fname), file_name=fname, mime="text/csv", use_container_width=True)

    st.subheader("Rolling-update architecture")
    st.markdown("""
- **Flat persistent files:** add `live_contracts.csv`, `live_player_seasons.csv`, `live_scouting.csv`, and `live_suspensions.csv` at the repository root. The app automatically merges them with or substitutes for demo records.
- **Advanced-stat layer:** validated Baseball Savant/FanGraphs exports can be transformed into `live_player_seasons.csv`; metric availability is era-aware, so comparisons use only fields available for both player and precedent.
- **Contracts layer:** each row should be a contract-event with a frozen snapshot of information available on the signing date, preventing hindsight leakage.
- **Scouting layer:** 20–80 grades include source and scouting date; their model weight decays with age and MLB service time.
- **Risk layer:** verified suspensions remain separate, including PED category, date, games and source.
- **Next phase:** once this flat build is live, the automated scheduled updater can be reintroduced without changing the valuation logic.
""")

elif page == "Methodology":
    st.title("Methodology & Product Roadmap")
    st.markdown("""
### What the MVP does
The valuation is an **ensemble of market precedents and fundamental projected performance**. Historical contracts are stored as contract-events, meaning the features attached to a precedent are the features available when that contract was signed—not what happened afterward.

### Era-aware similarity
The target data window begins in 2008. Advanced metrics are not uniformly available across the whole period, so similarity is computed only on fields shared by the target player and each precedent; weights are then renormalized. This lets 2008–2014 pitch-tracking-era contracts remain useful without pretending bat-speed or modern Statcast data existed then.

### Scouting grades
For young players, 20–80 scouting grades add information that MLB performance may not yet capture. Their influence declines automatically with age and service time. Hitters support Hit / Power / Run / Arm / Field / Overall. Pitchers support Fastball / Slider / Curveball / Changeup / Control / Overall.

### Upside & regression
The projection engine identifies historical seasons from players at similar ages, positions/roles and current performance levels. It uses observed year-over-year WAR changes from that cohort inside a Monte Carlo framework, blended with a generic aging prior when the cohort is thin. Injury history affects expected availability.

### Off-field / suspension risk
The MVP treats verified suspensions as a discrete risk factor, with PED suspensions separated from other suspensions. The analyst can add a small documented qualitative adjustment, but there is no opaque “character score.”

### Market normalization
Rather than relying only on CPI, each historical contract is ranked within its signing-year market. Its AAV and guarantee percentile are mapped into the recent/target market. This is designed to preserve whether a deal was ordinary, upper-tier or market-setting in its own era.

### Backtesting roadmap
Once validated 2008–present contract-event data is loaded, the next calibration layer should repeatedly “rewind” to each offseason, fit only on prior information, predict the upcoming deals, and optimize blend weights based on AAV / total guarantee / term error. That is the path from MVP heuristics to a defensible negotiation model.
""")
    st.info("Data-source terms and licensing should be reviewed before commercial launch. The architecture intentionally supports official APIs, manual/licensed CSV imports, and source attribution rather than requiring brittle scraping of restricted sites.")
