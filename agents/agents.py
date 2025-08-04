# agents/agents.py
from __future__ import annotations
import os, json, math
from dataclasses import dataclass
from typing import TypedDict, Annotated, List, Dict
from operator import add

import pandas as pd
import numpy as np

from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from typing_extensions import Annotated as Ann

# -----------------------------
# Config & OpenAI client
# -----------------------------
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # small, fast
client = OpenAI()  # expects OPENAI_API_KEY in env

# -----------------------------
# Graph state
# -----------------------------
class State(TypedDict):
    facts_path: str                  # parquet with day, segment_key, clicks, conv, spent
    plan_path: str                   # plan.csv from your TS loop (day, segment_key, share, planned_dollars)
    window_days: int                 # trend window (e.g., 7)
    cap: float                       # pacing cap (e.g., 0.30)
    aov: float | None                # average order value (optional, for ROAS commentary)

    # outputs
    insights: Ann[List[dict], add]   # list of dicts from Trends agent
    plan_suggestions: Ann[List[dict], add]  # list of dicts from Planner agent
    rationale: str                   # human-friendly explanation from Planner agent

# -----------------------------
# Small helpers
# -----------------------------
def _metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "impressions" in df.columns and "clicks" in df.columns:
        df["ctr"] = np.where(df["impressions"] > 0, df["clicks"] / df["impressions"], np.nan)
    else:
        df["ctr"] = np.nan
    df["cpc"] = np.where(df["clicks"].gt(0), df["spent"] / df["clicks"], np.nan)
    df["cvr"] = np.where(df["clicks"].gt(0), df["conv"] / df["clicks"], np.nan)
    df["cpa"] = np.where(df["conv"].gt(0),  df["spent"] / df["conv"], np.nan)
    return df

def _week_split(facts: pd.DataFrame, end_day: pd.Timestamp, k: int):
    recent = facts.loc[(facts["day"] <= end_day) & (facts["day"] > end_day - pd.Timedelta(days=k))]
    prior  = facts.loc[(facts["day"] <= end_day - pd.Timedelta(days=k)) & (facts["day"] > end_day - pd.Timedelta(days=2*k))]
    return recent, prior

def _pct(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0: return np.nan
    return (a - b) / abs(b)

def _json_from_llm(system, user) -> dict:
    """Call OpenAI chat with JSON output."""
    resp = client.chat.completions.create(  # Chat Completions w/ JSON mode
        model=MODEL,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    txt = resp.choices[0].message.content
    try:
        return json.loads(txt)
    except Exception:
        return {"raw": txt}  # fall back if model returns plain text

# -----------------------------
# Node 1: Trends & Anomalies Agent
# -----------------------------
def trends_node(state: State) -> dict:
    facts = pd.read_parquet(state["facts_path"])
    facts["day"] = pd.to_datetime(facts["day"])
    end_day = facts["day"].max()
    k = int(state["window_days"])

    # Aggregate to segment-day if not already
    cols = ["day","segment_key","clicks","conv","spent"]
    facts = facts[cols].groupby(["day","segment_key"], as_index=False).sum()

    # Compute windowed summaries per segment
    recent, prior = _week_split(facts, end_day, k)
    num_cols = recent.select_dtypes(include="number").columns
    r = _metrics(recent.groupby("segment_key", as_index=False)[num_cols].sum())
    p = _metrics(prior.groupby("segment_key",  as_index=False)[num_cols].sum())

    merged = r.merge(p, on="segment_key", suffixes=("_r","_p"), how="left")

    # % deltas
    for m in ["ctr","cpc","cvr","cpa","clicks","conv","spent"]:
        merged[f"d_{m}"] = merged[f"{m}_r"] - merged.get(f"{m}_p")
        merged[f"pct_{m}"] = merged.apply(lambda x: _pct(x[f"{m}_r"], x.get(f"{m}_p")), axis=1)

    # Simple anomalies
    #  - conv > clicks on any day (attribution)
    #  - CPC spikes (> 2x prior)
    anomalies = []
    day_level = _metrics(facts)
    bad_days = day_level.loc[(day_level["conv"] > day_level["clicks"])]
    for _, rrow in bad_days.head(10).iterrows():
        anomalies.append({
            "segment_key": rrow["segment_key"],
            "day": str(rrow["day"].date()),
            "issue": "conversions_exceed_clicks",
            "note": "Likely view-through attribution; clamp for per-click modeling."
        })
    merged["cpc_spike"] = merged.apply(lambda x: (pd.notna(x.get("pct_cpc")) and x["pct_cpc"] > 1.0), axis=1)
    for _, row in merged.loc[merged["cpc_spike"]].head(10).iterrows():
        anomalies.append({
            "segment_key": row["segment_key"],
            "day": str(end_day.date()),
            "issue": "cpc_spike",
            "note": "CPC > 2x prior window; inspect auction/creative/targeting."
        })

    # Top movers (by |pct_cvr| and |pct_cpc|)
    movers = []
    top_cvr = merged.loc[merged["pct_cvr"].abs().sort_values(ascending=False).index].head(5)
    for _, t in top_cvr.iterrows():
        movers.append({
            "segment_key": t["segment_key"],
            "cvr_recent": _safe_float(t["cvr_r"]),
            "cvr_prior":  _safe_float(t.get("cvr_p")),
            "pct_cvr_change": _safe_float(t["pct_cvr"]),
            "clicks_recent": int(t.get("clicks_r", 0)),
            "conv_recent": int(t.get("conv_r", 0)),
        })

    # Ask the model to turn these into a crisp JSON insight list
    system = "You are a performance marketing analyst. Output concise JSON."
    user = json.dumps({
        "task": "summarize_trends",
        "window_days": k,
        "end_day": str(end_day.date()),
        "top_cvr_movers": movers,
        "anomalies": anomalies
    })
    llm_out = _json_from_llm(system, user)

    insights = llm_out.get("insights", [])
    if not insights:
        # Fallback: create simple bullets
        for m in movers:
            insights.append({
                "headline": f"{m['segment_key']} CVR change {pct(m['pct_cvr_change'])}",
                "evidence": {"clicks_recent": m["clicks_recent"], "conv_recent": m["conv_recent"]},
                "action_hint": "Monitor; if sustained, consider budget nudge."
            })
        insights += [{"headline": f"Anomaly: {a['issue']} on {a['segment_key']}",
                      "evidence": a, "action_hint": "Explain to stakeholders; apply per-click clamp in modeling."}
                     for a in anomalies[:3]]

    return {"insights": insights}

def _safe_float(x):
    try:
        return float(x) if x is not None and not (isinstance(x, float) and math.isnan(x)) else None
    except Exception:
        return None

def pct(x):
    return f"{x*100:.0f}%" if x is not None and not np.isnan(x) else "n/a"

# -----------------------------
# Node 2: Planner Agent (budget nudges)
# -----------------------------
def planner_node(state: State) -> dict:
    facts = pd.read_parquet(state["facts_path"])
    facts["day"] = pd.to_datetime(facts["day"])

    plan = pd.read_csv(state["plan_path"])
    plan["day"] = pd.to_datetime(plan["day"])

    cap = float(state["cap"])
    aov_in = state.get("aov")
    aov_val = None if aov_in is None or (isinstance(aov_in, float) and np.isnan(aov_in)) else float(aov_in)

    # latest day in plan
    d = plan["day"].max()
    today_plan = plan[plan["day"] == d].copy()

    # ensure we have a 'share' column (fallback from dollars if needed)
    if "share" not in today_plan.columns:
        if "planned_dollars" in today_plan.columns:
            total = today_plan["planned_dollars"].sum() or 1.0
            today_plan["share"] = today_plan["planned_dollars"] / total
        elif "dollars" in today_plan.columns:
            total = today_plan["dollars"].sum() or 1.0
            today_plan["share"] = today_plan["dollars"] / total
        else:
            # default equal split
            n = max(1, len(today_plan))
            today_plan["share"] = 1.0 / n

    # recent performance window (<= end_day, last k days)
    k = int(state["window_days"])
    end_day = facts["day"].max()
    recent = facts.loc[(facts["day"] <= end_day) & (facts["day"] > end_day -pd.Timedelta(days=k))]
    perf = recent.groupby("segment_key", as_index=False)[["clicks", "conv", "spent"]].sum()
    perf["cvr"] = np.where(perf["clicks"].gt(0), perf["conv"] / perf["clicks"], np.nan)

    if aov_val is None:
        perf["roas"] = np.nan
        metric = "cvr"
    else:
        # avoid np.where(aov * Series, ...) when aov is None; compute only when numeric
        perf["roas"] = np.where(perf["spent"].gt(0), (aov_val * perf["conv"]) / perf["spent"], np.nan)
        metric = "roas"

    # explicit suffixes to avoid clicks/cvr collisions
    merged = today_plan.merge(perf, on="segment_key", how="left", suffixes=("_plan", "_hist"))

    # volume filter uses history clicks
    merged["volume"] = merged["clicks_hist"].fillna(0) if "clicks_hist" in merged.columns else 0
    eligible = merged[merged["volume"] >= 10].copy()

    # if not enough volume, no-op (keep shares)
    if eligible.empty:
        return {
            "plan_suggestions": [],
            "rationale": "Not enough recent volume to justify changes; keeping shares as-is within pacing guardrails."
        }

    # choose metric and build up/down sets
    eligible["_metric"] = eligible[metric].fillna(-np.inf if metric == "roas" else 0.0)
    up = eligible.sort_values("_metric", ascending=False).head(3)
    dn = eligible.sort_values("_metric", ascending=True).head(3)

    # current shares
    shares = today_plan.set_index("segment_key")["share"].to_dict()

    def bounded_new_share(seg, delta):
        old = float(shares.get(seg, 0.0))
        delta = float(np.clip(delta, -cap, +cap))
        return max(0.0, old + delta)

    suggestions = []
    for _, r in up.iterrows():
        old = float(shares.get(r["segment_key"], 0.0))
        new = bounded_new_share(r["segment_key"], +0.10)
        suggestions.append({
            "segment_key": r["segment_key"],
            "change": "+10%",
            "old_share": old,
            "new_share": new,
            "reason": f"High {metric.upper()} with adequate volume"
        })
    for _, r in dn.iterrows():
        old = float(shares.get(r["segment_key"], 0.0))
        new = bounded_new_share(r["segment_key"], -0.10)
        suggestions.append({
            "segment_key": r["segment_key"],
            "change": "-10%",
            "old_share": old,
            "new_share": new,
            "reason": f"Low {metric.upper()} with adequate volume"
        })

    # re-normalize to sum=1 while keeping other segments unchanged
    new_shares = shares.copy()
    for s in suggestions:
        new_shares[s["segment_key"]] = s["new_share"]
    total = sum(new_shares.values()) or 1.0
    for k_ in list(new_shares.keys()):
        new_shares[k_] = new_shares[k_] / total

    final_suggestions = []
    for s in suggestions:
        final_suggestions.append({
            **s,
            "final_share": new_shares[s["segment_key"]]
        })

    # concise stakeholder rationale via LLM (optional)
    system = "You are a pragmatic media planner. Output concise JSON."
    user = json.dumps({
        "task": "summarize_plan_nudges",
        "pacing_cap": cap,
        "objective": "ROAS" if aov_val is not None else "CONVERSIONS",
        "suggestions": final_suggestions,
        "note": "Respect pacing cap; small nudges; keep simple guardrails."
    })
    llm_out = _json_from_llm(system, user)
    rationale = llm_out.get("rationale") or "Rebalanced toward stronger segments within pacing cap; monitor tomorrow."

    return {"plan_suggestions": final_suggestions, "rationale": rationale}


# -----------------------------
# Graph builder
# -----------------------------
def build_graph() -> StateGraph:
    g = StateGraph(State)
    g.add_node("trends", trends_node)
    g.add_node("planner", planner_node)
    g.add_edge(START, "trends")
    g.add_edge("trends", "planner")
    g.add_edge("planner", END)
    return g.compile()
