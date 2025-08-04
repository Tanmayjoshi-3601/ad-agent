# run_agents.py
import json
from agents.agents import build_graph

CFG = dict(
    facts_path="data/facts_daily.parquet",
    plan_path="plan.csv",
    window_days=7,
    cap=0.30,
    aov=80.0  # set e.g. 80.0 to enable ROAS commentary
)

def main():
    graph = build_graph()
    out = graph.invoke(CFG)
    # Save both blocks for the UI to consume
    with open("insights.json","w") as f:
        json.dump(out.get("insights", []), f, indent=2)
    with open("plan_suggestions.json","w") as f:
        json.dump({
            "suggestions": out.get("plan_suggestions", []),
            "rationale": out.get("rationale", "")
        }, f, indent=2)
    print("Wrote insights.json and plan_suggestions.json")

if __name__ == "__main__":
    main()
