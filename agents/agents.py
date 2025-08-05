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

def _json_from_llm(system, user, examples=None) -> dict:
    """Call OpenAI chat with JSON output and optional few-shot examples."""
    messages = [{"role":"system","content":system}]
    
    # Add few-shot examples if provided
    if examples:
        for example in examples:
            messages.append({"role": "user", "content": example["input"]})
            messages.append({"role": "assistant", "content": example["output"]})
    
    messages.append({"role":"user","content":user})
    
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
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
    system = """You are a performance marketing analyst. Think step-by-step:

1. First, identify the most significant performance changes
2. Determine if changes are statistically meaningful (sufficient volume)
3. Classify each insight by urgency (immediate action vs monitoring)
4. Provide specific, actionable recommendations

Output your analysis as JSON with this structure:
{
  "insights": [
    {
      "headline": "Brief description",
      "evidence": {"key_metrics": "supporting_data"},
      "action_hint": "Specific recommendation",
      "urgency": "high|medium|low",
      "confidence": "high|medium|low"
    }
  ]
}"""
    
    user = json.dumps({
        "task": "Analyze the following marketing performance data and provide actionable insights",
        "analysis_window": f"{k} days ending {str(end_day.date())}",
        "performance_data": {
            "top_conversion_rate_changes": movers,
            "detected_anomalies": anomalies,
            "note": "Focus on changes with statistical significance and business impact"
        }
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
    floor = 0.01 # minimum share
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
    
    # Prepare segments_recent data for LLM and internal logic
    segments_recent = []
    for _, row in merged.iterrows():
        segments_recent.append({
            "segment_key": row["segment_key"],
            "cvr_recent": _safe_float(row.get("cvr")) if metric=="cvr" else None,
            "roas_recent": _safe_float(row.get("roas")) if metric=="roas" else None,
            "clicks_recent": int(row["volume"]),
            "share_plan": float(row.get("share", 0.0)),
            "metric_value": _safe_float(row.get(metric))
        })

    # Filter for segments with at least some activity
    segments_recent = [s for s in segments_recent if s["clicks_recent"] > 0]

    current_shares = {s["segment_key"]: s["share_plan"] for s in segments_recent}
    
    # Enhanced heuristic: be more proactive in generating suggestions
    final_suggestions = []

    # Sort segments by performance metric (CVR or ROAS)
    performance_scores = []
    for seg in segments_recent:
        segment_key = seg["segment_key"]
        current_share = seg["share_plan"]
        clicks = seg["clicks_recent"]
        cvr = seg["cvr_recent"]
        roas = seg["roas_recent"]
        
        target_metric = cvr if metric == "cvr" else roas
        
        if target_metric is not None:
             performance_scores.append({
                "segment_key": segment_key,
                "metric": target_metric,
                "clicks": clicks,
                "current_share": current_share,
                "cvr": cvr,
                "roas": roas
            })

    # Sort by performance metric descending
    performance_scores.sort(key=lambda x: x["metric"], reverse=True)

    if len(performance_scores) >= 2:
        # Always suggest reallocating from worst to best performers
        top_performers = performance_scores[:min(3, len(performance_scores)//2)]
        bottom_performers = performance_scores[-min(3, len(performance_scores)//2):]

        # Suggest increasing budget for top performers
        for seg in top_performers:
            if seg["clicks"] >= 5 and seg["current_share"] < cap - 0.03:
                new_share = min(seg["current_share"] * 1.25, cap)
                if new_share > seg["current_share"] + 0.01:  # meaningful change
                    final_suggestions.append({
                        "segment_key": seg["segment_key"],
                        "change": "increase",
                        "old_share": seg["current_share"],
                        "new_share": new_share,
                        "reason": f"Top performer: {seg['metric']:.3f} {'ROAS' if aov_val else 'CVR'}, {seg['clicks']} clicks"
                    })

        # Suggest decreasing budget for bottom performers  
        for seg in bottom_performers:
            if seg["current_share"] > floor + 0.03:
                new_share = max(seg["current_share"] * 0.75, floor)
                if seg["current_share"] - new_share > 0.01:  # meaningful change
                    final_suggestions.append({
                        "segment_key": seg["segment_key"],
                        "change": "decrease",
                        "old_share": seg["current_share"],
                        "new_share": new_share,
                        "reason": f"Underperforming: {seg['metric']:.3f} {'ROAS' if aov_val else 'CVR'}, reallocate to better segments"
                    })

        # If no suggestions yet, create balanced reallocation suggestions
        if not final_suggestions and len(performance_scores) >= 4:
            # Find segments with significant share differences vs performance
            for i, seg in enumerate(performance_scores[:len(performance_scores)//2]):
                if seg["clicks"] >= 3 and seg["current_share"] < 0.15:
                    final_suggestions.append({
                        "segment_key": seg["segment_key"],
                        "change": "test_increase",
                        "old_share": seg["current_share"],
                        "new_share": min(seg["current_share"] + 0.05, cap),
                        "reason": f"Test scaling: good performance ({seg['metric']:.3f}) with low share"
                    })
                    if len(final_suggestions) >= 2:
                        break

    # Normalize suggestions to ensure total share is 1.0
    if final_suggestions:
        current_total_share = sum(s["old_share"] for s in final_suggestions)
        adjustments = {}
        
        # Calculate net change from suggestions
        for s in final_suggestions:
            key = s["segment_key"]
            delta = s["new_share"] - s["old_share"]
            adjustments[key] = adjustments.get(key, 0) + delta
        
        # Distribute excess share/deficit to/from other segments
        unallocated_share = 1.0 - sum(s["new_share"] for s in final_suggestions)
        other_segments = [s for s in segments_recent if s["segment_key"] not in [sugg["segment_key"] for sugg in final_suggestions]]
        
        if other_segments:
            total_other_share = sum(s["share_plan"] for s in other_segments)
            if total_other_share > 0:
                for s in other_segments:
                    share_to_add = unallocated_share * (s["share_plan"] / total_other_share)
                    s["new_share"] = s["share_plan"] + share_to_add
                    s["new_share"] = max(floor, min(s["new_share"], cap)) # Apply bounds
        
        # Update final suggestions with potentially adjusted new_shares
        final_suggestions_updated = []
        for s in final_suggestions:
            s["new_share"] = max(floor, min(s["new_share"], cap))
            final_suggestions_updated.append(s)
        final_suggestions = final_suggestions_updated


    # Enhanced business-context prompting for stakeholder communication
    system = f"""You are a senior media strategist presenting to business stakeholders. 
    
Your role: Translate data insights into clear business recommendations.
Optimization goal: {"ROAS (Return on Ad Spend)" if aov_val is not None else "Conversion Volume"}
Risk tolerance: Conservative (max {cap*100:.0f}% daily changes)

Provide output in this JSON format:
{{
  "rationale": "2-3 sentence executive summary of recommended changes and expected impact",
  "business_context": "Why these changes make sense from a business perspective",
  "risk_assessment": "Potential risks and mitigation strategies",
  "success_metrics": "How to measure if changes are working"
}}"""
    
    user = json.dumps({
        "situation": f"Budget optimization results for {len(segments_recent)} customer segments",
        "performance_window": f"Last {k} days analysis",
        "optimization_metric": "ROAS" if aov_val is not None else "Conversions",
        "budget_constraints": {
            "daily_change_limit": f"{cap*100:.0f}%",
            "minimum_segment_budget": f"{floor*100:.0f}%"
        },
        "recommended_changes": final_suggestions,
        "context": "These recommendations come from Thompson Sampling optimization and trend analysis"
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