# day_runner.py
import pandas as pd
from bandit.ts import sample_thetas, allocate
from bandit.update import update
from sim.simulate import simulate_day

def run_one_day(facts_day: pd.DataFrame, arms, prev_share: dict | None,
                daily_budget: float, pacing_cap: float = 0.30, floor: float = 0.05):
    # 1) Thompson draw per arm (Beta posterior sample)
    samples = sample_thetas(arms)  # uses np.random.beta under the hood

    # 2) Probability-matching allocation + exploration floor + pacing cap
    share, dollars = allocate(samples, daily_budget, prev_share, cap=pacing_cap, floor=floor)

    # 3) Offline outcomes for this day (budget -> clicks -> conversions)
    outcomes = simulate_day(facts_day, dollars)

    # 4) Posterior update with today's aggregated outcomes
    df_out = pd.DataFrame(outcomes, columns=["segment_key", "clicks", "conv", "spent"])
    update(arms, df_out)

    return share, dollars, df_out  # keep share for pacing on the next day
