import numpy as np
def sample_thetas(arms,rng=None):
    """
    The core TS step: draw one plausible rate from each arm’s Beta posterior. This implements probability matching—an arm is favored in proportion to the chance it’s truly the best given your data.
    """

    # one posterior sample per arm:  θ̃_k ~ Beta(α_k, β_k)
    rng = rng or np.random.default_rng(42)
    return {k: rng.beta(v.alpha, v.beta) for k,v in arms.items()}

def allocate(samples, budget, prev_share = None, cap = 0.30,floor = 0.05):
    keys, vals = list(samples.keys()), np.array(list(samples.values()))
    # Probability matching: convert sampled values to a budget share
    share = vals / (vals.sum() + 1e-12)

    # Ensure some exploration so no arm is starved
    floor_each = floor / len(keys)
    share = np.maximum(share, floor_each); share /= share.sum()

    # Pacing: cap day-over-day change in share
    if prev_share:
        prev = np.array([prev_share.get(k, 1/len(keys)) for k in keys])
        delta = np.clip(share - prev, -cap, +cap)
        share = prev + delta
        share = np.clip(share, 1e-9, None); share /= share.sum()

    dollars = (share * budget).round(2)
    return dict(zip(keys, share)), dict(zip(keys, dollars))