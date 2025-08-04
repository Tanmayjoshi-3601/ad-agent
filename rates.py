# rates.py
import numpy as np, pandas as pd

def prepare_rates(facts: pd.DataFrame, a=0.5, b=0.5):
    df = facts.copy()

    # Clamp conv to clicks for a per-click model (avoid p>1 from attribution)
    df["conv"] = df[["conv","clicks"]].min(axis=1)

    # Segment-level median CPC using valid days
    valid = df[(df["clicks"] > 0) & (df["spent"] > 0)]
    seg_cpc_median = (valid["spent"] / valid["clicks"]).groupby(valid["segment_key"]).median()
    global_cpc_med = float((valid["spent"] / valid["clicks"]).median())

    # Smoothed CVR posterior mean: (conv + a) / (clicks + a + b)
    agg = df.groupby("segment_key")[["clicks","conv"]].sum()
    seg_cvr_post = ((agg["conv"] + a) / (agg["clicks"] + a + b)).to_dict()

    # Dicts for fast lookup
    cpc_map = seg_cpc_median.to_dict()
    return cpc_map, global_cpc_med, seg_cvr_post
