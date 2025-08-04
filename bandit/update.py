import numpy as np

def update(arms, df_day):  # df_day: columns segment_key, clicks, conv
    for _ , r in df_day.iterrows():
        a = arms[r["segment_key"]]

        # binomial batch update: α += successes, β += failures
        a.alpha += float(r["conv"])
        a.beta += float(max(r["clicks"] - r["conv"], 0.0))))
        