
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import asyncio
from typing import Dict, List

from bandit.state import init_arms_from_facts
from bandit.ts import sample_thetas, allocate
from bandit.update import update
from sim.simulate import simulate_day
from rates import prepare_rates
from agents.agents import build_graph

app = FastAPI(title="Thompson Sampling Dashboard")

# Global state for progress tracking
ts_progress = {"status": "idle", "current_day": 0, "total_days": 0, "logs": [], "error": None}
agent_progress = {"status": "idle", "step": "", "insights": [], "suggestions": {}, "error": None}
current_arms = None

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Thompson Sampling Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .controls { display: flex; gap: 10px; margin-bottom: 20px; }
            .btn { padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
            .btn-primary { background: #007bff; color: white; }
            .btn-success { background: #28a745; color: white; }
            .btn:disabled { background: #6c757d; cursor: not-allowed; }
            .progress-container { margin: 20px 0; }
            .progress-bar { width: 100%; background: #e9ecef; border-radius: 4px; height: 20px; overflow: hidden; }
            .progress-fill { height: 100%; background: #007bff; transition: width 0.3s ease; }
            .status { margin: 10px 0; padding: 10px; border-radius: 4px; }
            .status.running { background: #d1ecf1; color: #0c5460; }
            .status.idle { background: #d4edda; color: #155724; }
            .status.error { background: #f8d7da; color: #721c24; }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
            .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .full-width { grid-column: 1 / -1; }
            .log-container { max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 12px; background: #f8f9fa; padding: 10px; border-radius: 4px; }
            .insights-list { max-height: 300px; overflow-y: auto; }
            .insight-item { border-left: 4px solid #007bff; padding: 10px; margin: 10px 0; background: #f8f9fa; }
            .insight-headline { font-weight: bold; color: #007bff; }
            .insight-action { font-style: italic; color: #6c757d; margin-top: 5px; }
            .step-indicator { display: flex; align-items: center; gap: 10px; margin: 10px 0; }
            .step-circle { width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; }
            .step-pending { background: #e9ecef; color: #6c757d; }
            .step-running { background: #007bff; color: white; }
            .step-complete { background: #28a745; color: white; }
            .step-error { background: #dc3545; color: white; }
            .explanation-box { background: #e8f4f8; border: 1px solid #bee5eb; border-radius: 8px; padding: 15px; margin: 15px 0; }
            .explanation-title { font-weight: bold; color: #0c5460; margin-bottom: 10px; }
            .tooltip { position: relative; cursor: help; border-bottom: 1px dotted #007bff; }
            .tooltip .tooltiptext { visibility: hidden; width: 300px; background-color: #555; color: white; text-align: left; border-radius: 6px; padding: 10px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -150px; opacity: 0; transition: opacity 0.3s; font-size: 12px; line-height: 1.4; }
            .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
            .concept-box { background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 12px; margin: 10px 0; font-size: 14px; }
            .demo-note { background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px; padding: 15px; margin: 15px 0; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 Smart Budget Optimization Dashboard</h1>
                
                <div class="explanation-box">
                    <div class="explanation-title">What is this system doing?</div>
                    <p>This dashboard demonstrates an AI-powered marketing budget optimization system. It uses two main technologies:</p>
                    <ul>
                        <li><strong><span class="tooltip">Thompson Sampling<span class="tooltiptext">A smart algorithm that learns which customer segments perform best by trying different budget allocations and remembering what works</span></span>:</strong> An intelligent algorithm that learns which customer groups are most valuable by testing different budget amounts</li>
                        <li><strong><span class="tooltip">AI Agents<span class="tooltiptext">AI assistants that analyze data, find patterns, and make recommendations about where to spend marketing budget</span></span>:</strong> AI assistants that analyze the results and suggest improvements</li>
                    </ul>
                </div>

                <div class="demo-note">
                    <strong>🎯 Demo Purpose:</strong> This simulates 7 days of marketing campaigns across different customer segments. The system learns which segments convert better and automatically adjusts budget allocation to maximize results.
                </div>
                
                <div class="controls">
                    <button class="btn btn-primary" onclick="startTSLoop()" id="ts-btn">
                        🚀 Start Smart Optimization
                    </button>
                    <button class="btn btn-success" onclick="startAgents()" id="agent-btn">
                        🤖 Get AI Insights
                    </button>
                    <button class="btn" onclick="reset()">🔄 Reset Demo</button>
                </div>
            </div>
            
            <!-- Thompson Sampling Progress -->
            <div class="card full-width">
                <h3>🎯 Smart Budget Learning Process</h3>
                
                <div class="concept-box">
                    <strong>How it works:</strong> The system starts with basic assumptions about each customer segment, then learns from real performance data to get smarter about where to spend money.
                </div>
                
                <div class="step-indicator">
                    <div class="step-circle step-pending" id="ts-step1">1</div>
                    <span><span class="tooltip">Initialize Learning Models<span class="tooltiptext">Set up the mathematical models for each customer segment based on historical data. Think of this as giving the AI a starting point to learn from.</span></span></span>
                </div>
                <div class="step-indicator">
                    <div class="step-circle step-pending" id="ts-step2">2</div>
                    <span><span class="tooltip">Run Daily Learning & Optimization<span class="tooltiptext">Each day, the system: 1) Decides how much budget to give each segment, 2) Simulates the results, 3) Learns from what worked, 4) Gets smarter for tomorrow</span></span></span>
                </div>
                
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" id="ts-progress" style="width: 0%"></div>
                    </div>
                    <div id="ts-status" class="status idle">Ready to start the learning process</div>
                </div>
                
                <div class="explanation-box">
                    <div class="explanation-title">What you'll see:</div>
                    <p>The system will simulate 7 days of marketing campaigns. Each day it:</p>
                    <ol>
                        <li>Decides how much budget to allocate to each customer segment</li>
                        <li>Simulates running ads to those segments</li>
                        <li>Measures clicks and conversions (sales)</li>
                        <li>Updates its knowledge about which segments work best</li>
                        <li>Uses this learning to make better decisions tomorrow</li>
                    </ol>
                </div>
                
                <div class="log-container" id="ts-logs"></div>
            </div>

            <!-- Agent Progress -->
            <div class="card full-width">
                <h3>🤖 AI Business Analyst</h3>
                
                <div class="concept-box">
                    <strong>What this does:</strong> After the optimization runs, AI agents analyze the results like a human marketing analyst would - looking for trends, problems, and opportunities.
                </div>
                
                <div class="step-indicator">
                    <div class="step-circle step-pending" id="agent-step1">1</div>
                    <span><span class="tooltip">Performance Trend Analysis<span class="tooltiptext">The AI examines which customer segments improved or declined, identifies unusual patterns, and flags potential issues that need attention</span></span></span>
                </div>
                <div class="step-indicator">
                    <div class="step-circle step-pending" id="agent-step2">2</div>
                    <span><span class="tooltip">Strategic Budget Recommendations<span class="tooltiptext">Based on the trends analysis, the AI suggests specific budget changes to improve performance, explaining why each change makes business sense</span></span></span>
                </div>
                
                <div class="explanation-box">
                    <div class="explanation-title">What the AI Analyst will tell you:</div>
                    <ul>
                        <li><strong>Performance Insights:</strong> Which customer segments are trending up or down</li>
                        <li><strong>Problem Detection:</strong> Any unusual patterns that might indicate issues</li>
                        <li><strong>Budget Recommendations:</strong> Specific suggestions on where to increase or decrease spending</li>
                        <li><strong>Business Rationale:</strong> Plain-English explanations for why each change makes sense</li>
                    </ul>
                </div>
                
                <div id="agent-status" class="status idle">Waiting for optimization to complete first</div>
                <div class="log-container" id="agent-logs"></div>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h3>💰 Current Budget Split</h3>
                    <div class="concept-box">
                        Shows how your marketing budget is currently divided between different customer segments. Each slice represents a different group of customers.
                    </div>
                    <div id="allocation-chart"></div>
                </div>
                <div class="card">
                    <h3>📈 Performance Over Time</h3>
                    <div class="concept-box">
                        Track how many <span class="tooltip">conversions<span class="tooltiptext">A conversion is when someone takes a desired action, like making a purchase, signing up, or downloading something</span></span> each customer segment generated each day. Rising lines = improving performance.
                    </div>
                    <div id="performance-chart"></div>
                </div>
                <div class="card full-width">
                    <h3>🎯 Conversion Success Rates</h3>
                    <div class="concept-box">
                        Shows what percentage of people who clicked on ads actually converted (made a purchase, signed up, etc.). Higher bars = more effective customer segments.
                        <br><strong>Example:</strong> If 5% conversion rate means 5 out of every 100 people who clicked actually bought something.
                    </div>
                    <div id="cvr-chart"></div>
                </div>
                <div class="card">
                    <h3>🔍 AI-Generated Insights</h3>
                    <div class="concept-box">
                        The AI analyst's observations about what's happening with your campaigns. These are like having a data scientist point out important trends you should know about.
                    </div>
                    <div id="insights" class="insights-list"></div>
                </div>
                <div class="card">
                    <h3>💡 Recommended Changes</h3>
                    <div class="concept-box">
                        Specific suggestions from the AI about how to adjust your budget to get better results. Each suggestion explains what to change and why.
                    </div>
                    <div id="suggestions"></div>
                </div>
            </div>
        </div>

        <script>
            let progressInterval;

            async function startTSLoop() {
                const btn = document.getElementById('ts-btn');
                btn.disabled = true;
                btn.textContent = 'Running...';
                
                // Reset progress
                updateTSStep(1, 'running');
                updateTSProgress(0, 'Initializing arms...');
                
                try {
                    const response = await fetch('/start-ts', { method: 'POST' });
                    const result = await response.json();
                    
                    if (result.status === 'started') {
                        // Start polling for progress
                        progressInterval = setInterval(pollTSProgress, 1000);
                    }
                } catch (error) {
                    updateTSStatus('Error: ' + error.message, 'error');
                    btn.disabled = false;
                    btn.textContent = 'Start TS Loop';
                }
            }

            async function pollTSProgress() {
                try {
                    const response = await fetch('/ts-progress');
                    const progress = await response.json();
                    
                    if (progress.status === 'running') {
                        const percentage = progress.total_days > 0 ? 
                            (progress.current_day / progress.total_days) * 100 : 0;
                        updateTSProgress(percentage, `Day ${progress.current_day}/${progress.total_days}`);
                        updateTSLogs(progress.logs);
                        
                        if (progress.current_day > 0) {
                            updateTSStep(2, 'running');
                        }
                    } else if (progress.status === 'complete') {
                        clearInterval(progressInterval);
                        updateTSStep(2, 'complete');
                        updateTSProgress(100, 'Complete!');
                        updateTSLogs(progress.logs);
                        updateCharts();
                        
                        document.getElementById('ts-btn').disabled = false;
                        document.getElementById('ts-btn').textContent = 'Start TS Loop';
                        document.getElementById('agent-btn').disabled = false;
                        updateAgentStatus('Ready to run', 'idle');
                    } else if (progress.status === 'error') {
                        clearInterval(progressInterval);
                        updateTSStep(1, 'error');
                        updateTSStatus('Error: ' + progress.error, 'error');
                        document.getElementById('ts-btn').disabled = false;
                        document.getElementById('ts-btn').textContent = 'Start TS Loop';
                    }
                } catch (error) {
                    console.error('Progress polling error:', error);
                }
            }

            async function startAgents() {
                const btn = document.getElementById('agent-btn');
                btn.disabled = true;
                btn.textContent = 'Running...';
                
                updateAgentStep(1, 'running');
                updateAgentStatus('Running trends analysis...', 'running');
                
                try {
                    const response = await fetch('/start-agents', { method: 'POST' });
                    const result = await response.json();
                    
                    if (result.status === 'started') {
                        // Start polling for agent progress
                        const agentInterval = setInterval(async () => {
                            try {
                                const agentResponse = await fetch('/agent-progress');
                                const agentProgress = await agentResponse.json();
                                
                                if (agentProgress.status === 'trends_complete') {
                                    updateAgentStep(1, 'complete');
                                    updateAgentStep(2, 'running');
                                    updateAgentStatus('Running budget planning...', 'running');
                                    updateInsights(agentProgress.insights);
                                } else if (agentProgress.status === 'complete') {
                                    clearInterval(agentInterval);
                                    updateAgentStep(2, 'complete');
                                    updateAgentStatus('Complete!', 'idle');
                                    updateInsights(agentProgress.insights);
                                    updateSuggestions(agentProgress.suggestions);
                                    
                                    btn.disabled = false;
                                    btn.textContent = 'Run Agents';
                                } else if (agentProgress.status === 'error') {
                                    clearInterval(agentInterval);
                                    updateAgentStep(1, 'error');
                                    updateAgentStatus('Error: ' + agentProgress.error, 'error');
                                    btn.disabled = false;
                                    btn.textContent = 'Run Agents';
                                }
                            } catch (error) {
                                console.error('Agent progress error:', error);
                            }
                        }, 1000);
                    }
                } catch (error) {
                    updateAgentStatus('Error: ' + error.message, 'error');
                    btn.disabled = false;
                    btn.textContent = 'Run Agents';
                }
            }

            function updateTSStep(step, status) {
                const element = document.getElementById(`ts-step${step}`);
                element.className = `step-circle step-${status}`;
            }

            function updateAgentStep(step, status) {
                const element = document.getElementById(`agent-step${step}`);
                element.className = `step-circle step-${status}`;
            }

            function updateTSProgress(percentage, message) {
                document.getElementById('ts-progress').style.width = percentage + '%';
                updateTSStatus(message, percentage === 100 ? 'idle' : 'running');
            }

            function updateTSStatus(message, type) {
                const status = document.getElementById('ts-status');
                status.textContent = message;
                status.className = `status ${type}`;
            }

            function updateAgentStatus(message, type) {
                const status = document.getElementById('agent-status');
                status.textContent = message;
                status.className = `status ${type}`;
            }

            function updateTSLogs(logs) {
                const container = document.getElementById('ts-logs');
                container.innerHTML = logs.slice(-10).map(log => `<div>${log}</div>`).join('');
                container.scrollTop = container.scrollHeight;
            }

            async function updateCharts() {
                try {
                    const plan = await fetch('/api/plan').then(r => r.json());
                    
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
                } catch (error) {
                    console.error('Chart update error:', error);
                }
            }

            function updateInsights(insights) {
                const container = document.getElementById('insights');
                if (!insights || insights.length === 0) {
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
                if (!data || !data.suggestions) {
                    container.innerHTML = '<p>No suggestions available</p>';
                    return;
                }
                
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

            function reset() {
                // Reset all progress indicators
                for (let i = 1; i <= 2; i++) {
                    updateTSStep(i, 'pending');
                    updateAgentStep(i, 'pending');
                }
                updateTSProgress(0, 'Ready to start');
                updateAgentStatus('Waiting for TS completion', 'idle');
                document.getElementById('ts-logs').innerHTML = '';
                document.getElementById('agent-logs').innerHTML = '';
                document.getElementById('insights').innerHTML = '<p>No insights available</p>';
                document.getElementById('suggestions').innerHTML = '<p>No suggestions available</p>';
                
                // Clear charts
                Plotly.purge('allocation-chart');
                Plotly.purge('performance-chart');
                Plotly.purge('cvr-chart');
                
                // Re-enable buttons
                document.getElementById('ts-btn').disabled = false;
                document.getElementById('ts-btn').textContent = 'Start TS Loop';
                document.getElementById('agent-btn').disabled = true;
                
                clearInterval(progressInterval);
            }

            // Initialize
            reset();
        </script>
    </body>
    </html>
    """

@app.post("/start-ts")
async def start_thompson_sampling(background_tasks: BackgroundTasks):
    global ts_progress
    if ts_progress["status"] == "running":
        return {"status": "already_running"}
    
    ts_progress = {"status": "running", "current_day": 0, "total_days": 7, "logs": [], "error": None}
    background_tasks.add_task(execute_ts_loop_with_progress)
    return {"status": "started"}

@app.get("/ts-progress")
async def get_ts_progress():
    return ts_progress

@app.post("/start-agents")
async def start_agents_endpoint(background_tasks: BackgroundTasks):
    global agent_progress
    if agent_progress["status"] == "running":
        return {"status": "already_running"}
    
    agent_progress = {"status": "running", "step": "trends", "insights": [], "suggestions": {}, "error": None}
    background_tasks.add_task(execute_agents_with_progress)
    return {"status": "started"}

@app.get("/agent-progress")
async def get_agent_progress():
    return agent_progress

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

async def execute_ts_loop_with_progress():
    global ts_progress, current_arms
    
    try:
        ts_progress["logs"].append("Loading configuration...")
        await asyncio.sleep(0.1)  # Allow UI to update
        
        # Load configuration
        facts = pd.read_parquet("data/facts_daily.parquet")
        cpc_map, global_cpc_med, cvr_prior_map = prepare_rates(facts, a=0.5, b=0.5)
        days = sorted(facts["day"].unique())[:7]  # Run 7 days
        
        ts_progress["total_days"] = len(days)
        ts_progress["logs"].append(f"Initialized for {len(days)} days")
        
        # Initialize arms
        ts_progress["logs"].append("Initializing arms from historical data...")
        current_arms = init_arms_from_facts(facts, seed="day0")
        ts_progress["logs"].append(f"Initialized {len(current_arms)} arms")
        
        # Run simulation
        rng = np.random.default_rng(123)
        prev_share, logs = None, []
        budget = 200.0
        cap, floor = 0.30, 0.05
        
        for i, d in enumerate(days):
            ts_progress["current_day"] = i + 1
            ts_progress["logs"].append(f"Processing day {d}...")
            
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
            total_clicks, total_conv = 0, 0
            for k in sorted(share):
                clicks = int(df_out[df_out.segment_key == k]["clicks"].sum())
                conv = int(df_out[df_out.segment_key == k]["conv"].sum())
                total_clicks += clicks
                total_conv += conv
                logs.append({
                    "day": str(d), "segment_key": k, "share": share[k], "dollars": dollars[k],
                    "clicks": clicks, "conv": conv
                })
            
            ts_progress["logs"].append(f"Day {d}: {total_clicks} clicks, {total_conv} conversions")
            prev_share = share
            
            # Small delay to show progress
            await asyncio.sleep(0.5)
        
        # Save results
        pd.DataFrame(logs).to_csv("plan.csv", index=False)
        ts_progress["logs"].append("Saved results to plan.csv")
        ts_progress["status"] = "complete"
        
    except Exception as e:
        ts_progress["status"] = "error"
        ts_progress["error"] = str(e)
        ts_progress["logs"].append(f"Error: {e}")

async def execute_agents_with_progress():
    global agent_progress
    
    try:
        agent_progress["logs"] = ["Starting trends analysis..."]
        
        # Configure and run agents
        cfg = {
            "facts_path": "data/facts_daily.parquet",
            "plan_path": "plan.csv",
            "window_days": 7,
            "cap": 0.30,
            "aov": 80.0
        }
        
        graph = build_graph()
        
        # Run trends analysis
        agent_progress["step"] = "trends"
        agent_progress["logs"].append("Analyzing performance trends...")
        await asyncio.sleep(1)  # Simulate processing time
        
        result = graph.invoke(cfg)
        
        # Update progress after trends
        agent_progress["insights"] = result.get("insights", [])
        agent_progress["status"] = "trends_complete"
        agent_progress["logs"].append(f"Found {len(agent_progress['insights'])} insights")
        
        await asyncio.sleep(1)  # Allow UI to update
        
        # Budget planning step
        agent_progress["step"] = "planning"
        agent_progress["logs"].append("Generating budget recommendations...")
        await asyncio.sleep(1)  # Simulate processing time
        
        # Save results
        agent_progress["suggestions"] = {
            "suggestions": result.get("plan_suggestions", []),
            "rationale": result.get("rationale", "")
        }
        
        with open("insights.json", "w") as f:
            json.dump(agent_progress["insights"], f, indent=2)
        
        with open("plan_suggestions.json", "w") as f:
            json.dump(agent_progress["suggestions"], f, indent=2)
        
        agent_progress["status"] = "complete"
        agent_progress["logs"].append(f"Generated {len(agent_progress['suggestions']['suggestions'])} suggestions")
            
    except Exception as e:
        agent_progress["status"] = "error"
        agent_progress["error"] = str(e)
        agent_progress["logs"].append(f"Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
