# run_loop.py
import pandas as pd
import numpy as np

from bandit.state import init_arms_from_facts
from bandit.ts import sample_thetas, allocate
from bandit.update import update
from sim.simulate import simulate_day
from rates import prepare_rates



CFG = dict(
    path="data/facts_daily.parquet",
    days=7,
    budget=200.0,
    cap=0.30,
    floor=0.05,
    seed_mode="day0"   # or "history"
)

def main():
    facts = pd.read_parquet(CFG["path"])  # needs pyarrow installed
    cpc_map, global_cpc_med, cvr_prior_map = prepare_rates(facts, a=0.5, b=0.5)
    days  = sorted(facts["day"].unique())[:CFG["days"]]

    # Initialize Beta(α,β) per arm from the seed slice
    arms = init_arms_from_facts(facts, seed=CFG["seed_mode"])

    # reproducible RNG for this whole run
    rng = np.random.default_rng(123)

    prev_share, logs = None, []
    print(f"Running {len(days)} days, budget=${CFG['budget']}, cap={CFG['cap']}, floor={CFG['floor']}")

    for d in days:
        day_slice = facts.loc[facts["day"] == d, ["segment_key","clicks","conv","spent"]]

        # 1) Thompson draw per arm
        samples = sample_thetas(arms, rng=rng)

        # 2) Allocate with probability matching + floor + pacing
        share, dollars = allocate(samples, CFG["budget"], prev_share, cap=CFG["cap"], floor=CFG["floor"])

        # 3) Offline outcomes for this day
        outcomes = simulate_day(day_slice, dollars, rng=rng, arms=arms,
                        cpc_map=cpc_map, global_cpc_med=global_cpc_med,
                        cvr_prior_map=cvr_prior_map)
        df_out = pd.DataFrame(outcomes, columns=["segment_key","clicks","conv","spent"])

        # 4) Posterior update
        update(arms, df_out)

        # ---- logging & invariants ----
        total_share = sum(share.values())
        assert abs(total_share - 1.0) < 1e-8, f"Shares must sum to 1, got {total_share}"
        if prev_share:
            # pacing check
            for k in share:
                assert abs(share[k] - prev_share.get(k, 0)) <= CFG["cap"] + 1e-9, "Pacing cap breached"

        print(f"\nDay {d}:")
        for k in sorted(share):
            clicks = int(df_out[df_out.segment_key == k]["clicks"].sum())
            conv   = int(df_out[df_out.segment_key == k]["conv"].sum())
            print(f"  {k}: share={share[k]:.3f}, $={dollars[k]:.2f}, clicks={clicks}, conv={conv}")
            logs.append({
                "day": str(d), "segment_key": k, "share": share[k], "dollars": dollars[k],
                "clicks": clicks, "conv": conv
            })
        prev_share = share

    pd.DataFrame(logs).to_csv("plan.csv", index=False)
    print("\nWrote plan.csv")

if __name__ == "__main__":
    main()
