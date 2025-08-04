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
import pandas as pd
import numpy as np

def prepare_rates(facts, a=0.5, b=0.5):
    """
    Prepare CPC and CVR maps from historical facts data
    """
    # Calculate segment-level medians for CPC
    facts_with_metrics = facts.copy()
    facts_with_metrics['cpc'] = np.where(
        facts_with_metrics['clicks'] > 0,
        facts_with_metrics['spent'] / facts_with_metrics['clicks'],
        np.nan
    )
    facts_with_metrics['cvr'] = np.where(
        facts_with_metrics['clicks'] > 0,
        facts_with_metrics['conv'] / facts_with_metrics['clicks'],
        np.nan
    )
    
    # CPC map by segment
    cpc_map = facts_with_metrics.groupby('segment_key')['cpc'].median().to_dict()
    global_cpc_med = facts_with_metrics['cpc'].median()
    
    # CVR prior map by segment  
    cvr_prior_map = facts_with_metrics.groupby('segment_key')['cvr'].median().to_dict()
    
    return cpc_map, global_cpc_med, cvr_prior_map
