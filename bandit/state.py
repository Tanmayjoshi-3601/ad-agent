from dataclasses import dataclass
import pandas as pd

@dataclass
class Arm:
    key:str
    alpha: float = 1.0  # prior (and running) successes
    beta: float = 1.0 # prior (and running) failures

def init_arms_from_facts(facts: pd.DataFrame, seed = "day0"):
    """
    Build Beta(α,β) posteriors per arm from your processed facts table,
    seed = 'day0' uses the first day; seed+'history' uses all historical rows.
    """
    arms = {}
    if seed == "day0":
        start = facts["day"].min()
        it = facts[facts["day"] == start].groupby("segment_key")
    else:
        it = facts.groupby("segment_key")

    for k,g in it:
        conv = float(g["conv"].sum())
        clicks = float(g["clicks"].sum())
        arms[k] = Arm(k,1.0 + conv,1.0 + max(clicks-conv,0.0))

    return armsms
        