
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

from bandit.state import init_arms_from_facts
from bandit.ts import sample_thetas, allocate
from bandit.update import update
from sim.simulate import simulate_day
from rates import prepare_rates
from agents.agents import build_graph

app = FastAPI(title="Thompson Sampling Dashboard")

# Global state
current_arms = None
current_plan = []
current_insights = []
current_suggestions = []
simulation_running = False

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Thompson Sampling Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .controls { display: flex; gap: 10px; margin-bottom: 20px; }
            .btn { padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
            .btn-primary { background: #007bff; color: white; }
            .btn-success { background: #28a745; color: white; }
            .btn:disabled { background: #6c757d; cursor: not-allowed; }
            .status { margin: 10px 0; padding: 10px; border-radius: 4px; }
            .status.running { background: #d1ecf1; color: #0c5460; }
            .status.idle { background: #d4edda; color: #155724; }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
            .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .full-width { grid-column: 1 / -1; }
            .insights-list { max-height: 300px; overflow-y: auto; }
            .insight-item { border-left: 4px solid #007bff; padding: 10px; margin: 10px 0; background: #f8f9fa; }
            .insight-headline { font-weight: bold; color: #007bff; }
            .insight-action { font-style: italic; color: #6c757d; margin-top: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Thompson Sampling Dashboard</h1>
                <div class="controls">
                    <button class="btn btn-primary" onclick="runSimulation()">Run TS Loop</button>
                    <button class="btn btn-success" onclick="runAgents()">Run Agents</button>
                    <button class="btn" onclick="refreshData()">Refresh</button>
                </div>
                <div id="status" class="status idle">Ready</div>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h3>Budget Allocation</h3>
                    <div id="allocation-chart"></div>
                </div>
                <div class="card">
                    <h3>Performance Trends</h3>
                    <div id="performance-chart"></div>
                </div>
                <div class="card full-width">
                    <h3>Conversion Rates by Segment</h3>
                    <div id="cvr-chart"></div>
                </div>
                <div class="card">
                    <h3>Agent Insights</h3>
                    <div id="insights" class="insights-list"></div>
                </div>
                <div class="card">
                    <h3>Budget Suggestions</h3>
                    <div id="suggestions"></div>
                </div>
            </div>
        </div>

        <script>
            let refreshInterval;

            async function runSimulation() {
                updateStatus('Running Thompson Sampling...', 'running');
                try {
                    const response = await fetch('/run-ts', { method: 'POST' });
                    const result = await response.json();
                    updateStatus('TS Loop completed', 'idle');
                    refreshData();
                } catch (error) {
                    updateStatus('Error: ' + error.message, 'idle');
                }
            }

            async function runAgents() {
                updateStatus('Running agents...', 'running');
                try {
                    const response = await fetch('/run-agents', { method: 'POST' });
                    const result = await response.json();
                    updateStatus('Agents completed', 'idle');
                    refreshData();
                } catch (error) {
                    updateStatus('Error: ' + error.message, 'idle');
                }
            }

            async function refreshData() {
                try {
                    const [plan, insights, suggestions] = await Promise.all([
                        fetch('/api/plan').then(r => r.json()),
                        fetch('/api/insights').then(r => r.json()),
                        fetch('/api/suggestions').then(r => r.json())
                    ]);
                    
                    updateCharts(plan, insights, suggestions);
                    updateInsights(insights);
                    updateSuggestions(suggestions);
                } catch (error) {
                    console.error('Refresh error:', error);
                }
            }

            function updateStatus(message, type) {
                const status = document.getElementById('status');
                status.textContent = message;
                status.className = `status ${type}`;
            }

            function updateCharts(plan, insights, suggestions) {
                if (plan.length > 0) {
                    // Budget allocation pie chart
                    const latestDay = plan.reduce((latest, row) => 
                        row.day > latest ? row.day : latest, plan[0].day);
                    const latestPlan = plan.filter(row => row.day === latestDay);
                    
                    const pieData = [{
                        type: 'pie',
                        labels: latestPlan.map(row => row.segment_key),
                        values: latestPlan.map(row => row.dollars),
                        textinfo: 'label+percent',
                        textposition: 'outside'
                    }];
                    Plotly.newPlot('allocation-chart', pieData, {height: 300});

                    // Performance trends
                    const segments = [...new Set(plan.map(row => row.segment_key))];
                    const performanceData = segments.map(segment => {
                        const segmentData = plan.filter(row => row.segment_key === segment);
                        return {
                            x: segmentData.map(row => row.day),
                            y: segmentData.map(row => row.conv || 0),
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: segment,
                            line: {width: 2}
                        };
                    });
                    Plotly.newPlot('performance-chart', performanceData, {
                        height: 300,
                        xaxis: {title: 'Day'},
                        yaxis: {title: 'Conversions'}
                    });

                    // CVR by segment
                    const cvrData = latestPlan.map(row => ({
                        segment: row.segment_key,
                        cvr: row.clicks > 0 ? (row.conv / row.clicks) : 0,
                        clicks: row.clicks || 0
                    }));
                    
                    const barData = [{
                        x: cvrData.map(d => d.segment),
                        y: cvrData.map(d => d.cvr * 100),
                        type: 'bar',
                        marker: {color: 'rgba(0, 123, 255, 0.7)'},
                        text: cvrData.map(d => `${d.clicks} clicks`),
                        textposition: 'outside'
                    }];
                    Plotly.newPlot('cvr-chart', barData, {
                        height: 300,
                        xaxis: {title: 'Segment', tickangle: -45},
                        yaxis: {title: 'CVR (%)'}
                    });
                }
            }

            function updateInsights(insights) {
                const container = document.getElementById('insights');
                if (insights.length === 0) {
                    container.innerHTML = '<p>No insights available</p>';
                    return;
                }
                
                const html = insights.map(insight => `
                    <div class="insight-item">
                        <div class="insight-headline">${insight.headline || 'Insight'}</div>
                        <div class="insight-action">${insight.action_hint || ''}</div>
                    </div>
                `).join('');
                container.innerHTML = html;
            }

            function updateSuggestions(data) {
                const container = document.getElementById('suggestions');
                const suggestions = data.suggestions || [];
                
                if (suggestions.length === 0) {
                    container.innerHTML = `<p>${data.rationale || 'No suggestions available'}</p>`;
                    return;
                }
                
                const html = suggestions.map(s => `
                    <div class="insight-item">
                        <div class="insight-headline">${s.segment_key}: ${s.change}</div>
                        <div>Old: ${(s.old_share * 100).toFixed(1)}% → New: ${(s.new_share * 100).toFixed(1)}%</div>
                        <div class="insight-action">${s.reason}</div>
                    </div>
                `).join('');
                container.innerHTML = html + `<p style="margin-top: 15px; font-style: italic;">${data.rationale}</p>`;
            }

            // Auto-refresh every 30 seconds
            refreshInterval = setInterval(refreshData, 30000);
            
            // Initial load
            refreshData();
        </script>
    </body>
    </html>
    """

@app.post("/run-ts")
async def run_thompson_sampling(background_tasks: BackgroundTasks):
    global simulation_running
    if simulation_running:
        return {"status": "already_running"}
    
    background_tasks.add_task(execute_ts_loop)
    return {"status": "started"}

@app.post("/run-agents")
async def run_agents_endpoint(background_tasks: BackgroundTasks):
    background_tasks.add_task(execute_agents)
    return {"status": "started"}

@app.get("/api/plan")
async def get_plan():
    try:
        df = pd.read_csv("plan.csv")
        return df.to_dict('records')
    except:
        return []

@app.get("/api/insights")
async def get_insights():
    try:
        with open("insights.json", "r") as f:
            return json.load(f)
    except:
        return []

@app.get("/api/suggestions")
async def get_suggestions():
    try:
        with open("plan_suggestions.json", "r") as f:
            return json.load(f)
    except:
        return {"suggestions": [], "rationale": "No suggestions available"}

async def execute_ts_loop():
    global simulation_running, current_arms, current_plan
    simulation_running = True
    
    try:
        # Load configuration
        facts = pd.read_parquet("data/facts_daily.parquet")
        cpc_map, global_cpc_med, cvr_prior_map = prepare_rates(facts, a=0.5, b=0.5)
        days = sorted(facts["day"].unique())[:7]  # Run 7 days
        
        # Initialize arms
        current_arms = init_arms_from_facts(facts, seed="day0")
        
        # Run simulation
        rng = np.random.default_rng(123)
        prev_share, logs = None, []
        budget = 200.0
        cap, floor = 0.30, 0.05
        
        for d in days:
            day_slice = facts.loc[facts["day"] == d, ["segment_key","clicks","conv","spent"]]
            
            # Thompson sampling
            samples = sample_thetas(current_arms, rng=rng)
            share, dollars = allocate(samples, budget, prev_share, cap=cap, floor=floor)
            
            # Simulate outcomes
            outcomes = simulate_day(day_slice, dollars, rng=rng, arms=current_arms,
                            cpc_map=cpc_map, global_cpc_med=global_cpc_med,
                            cvr_prior_map=cvr_prior_map)
            df_out = pd.DataFrame(outcomes, columns=["segment_key","clicks","conv","spent"])
            
            # Update posteriors
            update(current_arms, df_out)
            
            # Log results
            for k in sorted(share):
                clicks = int(df_out[df_out.segment_key == k]["clicks"].sum())
                conv = int(df_out[df_out.segment_key == k]["conv"].sum())
                logs.append({
                    "day": str(d), "segment_key": k, "share": share[k], "dollars": dollars[k],
                    "clicks": clicks, "conv": conv
                })
            prev_share = share
        
        # Save results
        pd.DataFrame(logs).to_csv("plan.csv", index=False)
        current_plan = logs
        
    except Exception as e:
        print(f"TS Loop error: {e}")
    finally:
        simulation_running = False

async def execute_agents():
    global current_insights, current_suggestions
    
    try:
        # Configure and run agents
        cfg = {
            "facts_path": "data/facts_daily.parquet",
            "plan_path": "plan.csv",
            "window_days": 7,
            "cap": 0.30,
            "aov": 80.0
        }
        
        graph = build_graph()
        result = graph.invoke(cfg)
        
        # Save results
        current_insights = result.get("insights", [])
        current_suggestions = {
            "suggestions": result.get("plan_suggestions", []),
            "rationale": result.get("rationale", "")
        }
        
        with open("insights.json", "w") as f:
            json.dump(current_insights, f, indent=2)
        
        with open("plan_suggestions.json", "w") as f:
            json.dump(current_suggestions, f, indent=2)
            
    except Exception as e:
        print(f"Agents error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
