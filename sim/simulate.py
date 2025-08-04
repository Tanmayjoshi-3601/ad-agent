# sim/simulate.py
import numpy as np

def simulate_day(facts_day, dollars_by_key, rng=None, arms=None, cpc_map=None, global_cpc_med=0.5,
                 cvr_prior_map=None, cpc_floor=None, cvr_cap=0.5):
    rng = rng or np.random.default_rng(0)
    if cpc_floor is None:
        cpc_floor = max(0.10, 0.25 * global_cpc_med)  # data-backed floor

    out = []
    for key, dollars in dollars_by_key.items():
        row = facts_day[facts_day["segment_key"] == key]

        # Observed (may be NaN or zero)
        if not row.empty:
            clicks_hist = float(row["clicks"].values[0])
            conv_hist   = float(row["conv"].values[0])
            spent_hist  = float(row["spent"].values[0])
            cpc_obs = spent_hist / max(clicks_hist, 1.0)
            cvr_obs = conv_hist   / max(clicks_hist, 1.0)
        else:
            cpc_obs = np.nan
            cvr_obs = np.nan

        # CPC: prefer observed > seg median > global median; enforce floor
        cpc_hat = cpc_map.get(key, np.nan)
        cpc = cpc_obs if (not np.isnan(cpc_obs) and cpc_obs > 0) else (cpc_hat if not np.isnan(cpc_hat) else global_cpc_med)
        cpc = max(cpc, cpc_floor)

        # CVR: posterior mean from arms, blended with smoothed historical prior
        cvr_post = arms[key].alpha / (arms[key].alpha + arms[key].beta)
        cvr_hist = cvr_prior_map.get(key, cvr_post)
        if not np.isnan(cvr_obs) and cvr_obs >= 0:
            cvr = 0.5 * cvr_post + 0.5 * cvr_obs
        else:
            cvr = 0.7 * cvr_post + 0.3 * cvr_hist
        cvr = float(np.clip(cvr, 1e-5, cvr_cap))

        clicks = int(dollars / cpc)
        conv   = int(rng.binomial(n=max(clicks, 0), p=cvr))
        out.append((key, clicks, conv, dollars))
    return out
