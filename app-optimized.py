import json
import os
from datetime import datetime, timedelta
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
import asyncio
from typing import Dict, List

app = FastAPI(title="Thompson Sampling Dashboard")

# Global state for progress tracking
ts_progress = {"status": "idle", "current_day": 0, "total_days": 0, "logs": [], "error": None}
agent_progress = {"status": "idle", "step": "", "insights": [], "suggestions": {}, "error": None}
current_arms = None

# Lazy imports to reduce initial bundle size
def get_pandas():
    import pandas as pd
    return pd

def get_numpy():
    import numpy as np
    return np

def get_plotly():
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.utils import PlotlyJSONEncoder
    return go, px, PlotlyJSONEncoder

def get_bandit_modules():
    from bandit.state import init_arms_from_facts
    from bandit.ts import sample_thetas, allocate
    from bandit.update import update
    return init_arms_from_facts, sample_thetas, allocate, update

def get_sim_module():
    from sim.simulate import simulate_day
    return simulate_day

def get_rates_module():
    from rates import prepare_rates
    return prepare_rates

def get_agents_module():
    from agents.agents import build_graph
    return build_graph

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Thompson Sampling Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
        <style>
            :root {
                --bg-primary: #0d1117;
                --bg-secondary: #161b22;
                --bg-tertiary: #1f2937;
                --accent-primary: #58a6ff;
                --accent-secondary: #7c3aed;
                --accent-success: #10b981;
                --accent-warning: #f59e0b;
                --accent-error: #ef4444;
                --text-primary: #f0f6fc;
                --text-secondary: #8b949e;
                --text-muted: #6e7681;
                --border-primary: #30363d;
                --border-secondary: #21262d;
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: var(--bg-primary);
                color: var(--text-primary);
                line-height: 1.6;
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 2rem;
            }
            
            .header {
                text-align: center;
                margin-bottom: 3rem;
            }
            
            .header h1 {
                font-size: 2.5rem;
                margin-bottom: 0.5rem;
                background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .header p {
                color: var(--text-secondary);
                font-size: 1.1rem;
            }
            
            .controls {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 2rem;
                margin-bottom: 3rem;
            }
            
            .control-panel {
                background: var(--bg-secondary);
                border: 1px solid var(--border-primary);
                border-radius: 12px;
                padding: 2rem;
            }
            
            .control-panel h2 {
                color: var(--accent-primary);
                margin-bottom: 1.5rem;
                font-size: 1.3rem;
            }
            
            .form-group {
                margin-bottom: 1.5rem;
            }
            
            .form-group label {
                display: block;
                color: var(--text-secondary);
                margin-bottom: 0.5rem;
                font-weight: 500;
            }
            
            .form-group input, .form-group select {
                width: 100%;
                padding: 0.75rem;
                background: var(--bg-tertiary);
                border: 1px solid var(--border-primary);
                border-radius: 6px;
                color: var(--text-primary);
                font-size: 0.9rem;
            }
            
            .form-group input:focus, .form-group select:focus {
                outline: none;
                border-color: var(--accent-primary);
                box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.1);
            }
            
            .btn {
                background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
                color: white;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 6px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
                width: 100%;
                margin-bottom: 1rem;
            }
            
            .btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(88, 166, 255, 0.3);
            }
            
            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .status-card {
                background: var(--bg-secondary);
                border: 1px solid var(--border-primary);
                border-radius: 12px;
                padding: 2rem;
                margin-bottom: 2rem;
            }
            
            .status-card h2 {
                color: var(--accent-primary);
                margin-bottom: 1rem;
                font-size: 1.5rem;
            }
            
            .status {
                display: inline-block;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 0.8rem;
                letter-spacing: 0.5px;
                margin-bottom: 1rem;
            }
            
            .status.idle {
                background: var(--bg-tertiary);
                color: var(--text-muted);
            }
            
            .status.running {
                background: rgba(16, 185, 129, 0.2);
                color: var(--accent-success);
            }
            
            .status.error {
                background: rgba(239, 68, 68, 0.2);
                color: var(--accent-error);
            }
            
            .logs {
                background: var(--bg-tertiary);
                border: 1px solid var(--border-secondary);
                border-radius: 6px;
                padding: 1rem;
                max-height: 200px;
                overflow-y: auto;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 0.8rem;
                color: var(--text-secondary);
            }
            
            .log-entry {
                margin-bottom: 0.25rem;
            }
            
            .log-entry.error {
                color: var(--accent-error);
            }
            
            .log-entry.success {
                color: var(--accent-success);
            }
            
            .results {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 2rem;
                margin-top: 2rem;
            }
            
            .result-card {
                background: var(--bg-secondary);
                border: 1px solid var(--border-primary);
                border-radius: 12px;
                padding: 1.5rem;
            }
            
            .result-card h3 {
                color: var(--accent-primary);
                margin-bottom: 1rem;
            }
            
            .chart-container {
                width: 100%;
                height: 300px;
                background: var(--bg-tertiary);
                border-radius: 6px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: var(--text-muted);
            }
            
            .info-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 1rem;
                margin-top: 1rem;
            }
            
            .info-item {
                background: var(--bg-tertiary);
                padding: 1rem;
                border-radius: 6px;
                border: 1px solid var(--border-secondary);
                text-align: center;
            }
            
            .info-label {
                color: var(--text-muted);
                font-size: 0.8rem;
                margin-bottom: 0.25rem;
            }
            
            .info-value {
                color: var(--text-primary);
                font-weight: 600;
                font-size: 1.1rem;
            }
            
            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid var(--border-primary);
                border-radius: 50%;
                border-top-color: var(--accent-primary);
                animation: spin 1s ease-in-out infinite;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            
            .hidden {
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Thompson Sampling Dashboard</h1>
                <p>AI-Powered Ad Budget Optimization with Multi-Armed Bandit</p>
            </div>
            
            <div class="controls">
                <div class="control-panel">
                    <h2>Thompson Sampling</h2>
                    <div class="form-group">
                        <label for="horizon">Horizon (days)</label>
                        <input type="number" id="horizon" value="7" min="1" max="30">
                    </div>
                    <div class="form-group">
                        <label for="budget">Daily Budget ($)</label>
                        <input type="number" id="budget" value="200" min="10" step="10">
                    </div>
                    <div class="form-group">
                        <label for="aov">Average Order Value ($)</label>
                        <input type="number" id="aov" value="80" min="1" step="1">
                    </div>
                    <button class="btn" onclick="runThompsonSampling()">
                        <span id="ts-btn-text">Run Thompson Sampling</span>
                        <span id="ts-loading" class="loading hidden"></span>
                    </button>
                </div>
                
                <div class="control-panel">
                    <h2>AI Agents</h2>
                    <div class="form-group">
                        <label for="window">Trend Window (days)</label>
                        <input type="number" id="window" value="7" min="1" max="30">
                    </div>
                    <div class="form-group">
                        <label for="cap">Pacing Cap</label>
                        <input type="number" id="cap" value="0.30" min="0.01" max="1" step="0.01">
                    </div>
                    <button class="btn" onclick="runAgents()">
                        <span id="agent-btn-text">Run AI Analysis</span>
                        <span id="agent-loading" class="loading hidden"></span>
                    </button>
                </div>
            </div>
            
            <div class="status-card">
                <h2>Thompson Sampling Status</h2>
                <div class="status idle" id="ts-status">Idle</div>
                <div class="logs" id="ts-logs">Ready to run Thompson Sampling...</div>
            </div>
            
            <div class="status-card">
                <h2>AI Agents Status</h2>
                <div class="status idle" id="agent-status">Idle</div>
                <div class="logs" id="agent-logs">Ready to run AI analysis...</div>
            </div>
            
            <div class="results" id="results" style="display: none;">
                <div class="result-card">
                    <h3>Allocation Results</h3>
                    <div class="chart-container" id="allocation-chart">
                        Chart will appear here
                    </div>
                </div>
                
                <div class="result-card">
                    <h3>Performance Metrics</h3>
                    <div class="info-grid" id="metrics-grid">
                        <!-- Metrics will be populated here -->
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            async function runThompsonSampling() {
                const btn = document.querySelector('#ts-btn-text');
                const loading = document.querySelector('#ts-loading');
                const status = document.querySelector('#ts-status');
                const logs = document.querySelector('#ts-logs');
                
                btn.classList.add('hidden');
                loading.classList.remove('hidden');
                status.textContent = 'Running';
                status.className = 'status running';
                logs.innerHTML = 'Starting Thompson Sampling...';
                
                const horizon = document.getElementById('horizon').value;
                const budget = document.getElementById('budget').value;
                const aov = document.getElementById('aov').value;
                
                try {
                    const response = await fetch('/api/run-thompson-sampling', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            horizon_days: parseInt(horizon),
                            daily_budget: parseFloat(budget),
                            aov: parseFloat(aov)
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error('Failed to run Thompson Sampling');
                    }
                    
                    const result = await response.json();
                    logs.innerHTML = 'Thompson Sampling completed successfully!';
                    status.textContent = 'Completed';
                    status.className = 'status success';
                    
                    // Update results
                    updateResults(result);
                    
                } catch (error) {
                    logs.innerHTML = `Error: ${error.message}`;
                    status.textContent = 'Error';
                    status.className = 'status error';
                } finally {
                    btn.classList.remove('hidden');
                    loading.classList.add('hidden');
                }
            }
            
            async function runAgents() {
                const btn = document.querySelector('#agent-btn-text');
                const loading = document.querySelector('#agent-loading');
                const status = document.querySelector('#agent-status');
                const logs = document.querySelector('#agent-logs');
                
                btn.classList.add('hidden');
                loading.classList.remove('hidden');
                status.textContent = 'Running';
                status.className = 'status running';
                logs.innerHTML = 'Starting AI analysis...';
                
                const window = document.getElementById('window').value;
                const cap = document.getElementById('cap').value;
                
                try {
                    const response = await fetch('/api/run-agents', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            window_days: parseInt(window),
                            cap: parseFloat(cap)
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error('Failed to run AI agents');
                    }
                    
                    const result = await response.json();
                    logs.innerHTML = 'AI analysis completed successfully!';
                    status.textContent = 'Completed';
                    status.className = 'status success';
                    
                } catch (error) {
                    logs.innerHTML = `Error: ${error.message}`;
                    status.textContent = 'Error';
                    status.className = 'status error';
                } finally {
                    btn.classList.remove('hidden');
                    loading.classList.add('hidden');
                }
            }
            
            function updateResults(result) {
                const resultsDiv = document.getElementById('results');
                resultsDiv.style.display = 'grid';
                
                // Update metrics
                const metricsGrid = document.getElementById('metrics-grid');
                metricsGrid.innerHTML = `
                    <div class="info-item">
                        <div class="info-label">Total Budget</div>
                        <div class="info-value">$${result.total_budget || 0}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Expected Conversions</div>
                        <div class="info-value">${result.expected_conversions || 0}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">ROI</div>
                        <div class="info-value">${result.roi || '0%'}</div>
                    </div>
                `;
            }
            
            // Poll for status updates
            setInterval(async () => {
                try {
                    const [tsResponse, agentResponse] = await Promise.all([
                        fetch('/api/ts-progress'),
                        fetch('/api/agent-progress')
                    ]);
                    
                    if (tsResponse.ok) {
                        const tsData = await tsResponse.json();
                        updateStatus('ts', tsData);
                    }
                    
                    if (agentResponse.ok) {
                        const agentData = await agentResponse.json();
                        updateStatus('agent', agentData);
                    }
                } catch (error) {
                    console.error('Error polling status:', error);
                }
            }, 2000);
            
            function updateStatus(type, data) {
                const status = document.querySelector(`#${type}-status`);
                const logs = document.querySelector(`#${type}-logs`);
                
                status.textContent = data.status;
                status.className = `status ${data.status}`;
                
                if (data.logs && data.logs.length > 0) {
                    logs.innerHTML = data.logs.map(log => 
                        `<div class="log-entry ${log.includes('Error') ? 'error' : 'success'}">${log}</div>`
                    ).join('');
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/api/ts-progress")
def get_ts_progress():
    return ts_progress

@app.get("/api/agent-progress")
def get_agent_progress():
    return agent_progress

@app.post("/api/run-thompson-sampling")
async def run_thompson_sampling(params: dict):
    try:
        # Lazy import heavy modules
        pd = get_pandas()
        np = get_numpy()
        go, px, PlotlyJSONEncoder = get_plotly()
        init_arms_from_facts, sample_thetas, allocate, update = get_bandit_modules()
        simulate_day = get_sim_module()
        prepare_rates = get_rates_module()
        
        # Your existing Thompson Sampling logic here
        # This is a simplified version - you'll need to adapt your full logic
        
        ts_progress["status"] = "running"
        ts_progress["logs"] = ["Starting Thompson Sampling..."]
        
        # Simulate the process
        await asyncio.sleep(1)
        
        ts_progress["logs"].append("Loading data...")
        await asyncio.sleep(1)
        
        ts_progress["logs"].append("Running simulation...")
        await asyncio.sleep(2)
        
        ts_progress["status"] = "completed"
        ts_progress["logs"].append("Thompson Sampling completed!")
        
        return {
            "status": "success",
            "total_budget": params.get("daily_budget", 200) * params.get("horizon_days", 7),
            "expected_conversions": 25,
            "roi": "15%"
        }
        
    except Exception as e:
        ts_progress["status"] = "error"
        ts_progress["error"] = str(e)
        ts_progress["logs"].append(f"Error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/run-agents")
async def run_agents(params: dict):
    try:
        # Lazy import heavy modules
        build_graph = get_agents_module()
        
        agent_progress["status"] = "running"
        agent_progress["logs"] = ["Starting AI analysis..."]
        
        # Simulate the process
        await asyncio.sleep(1)
        
        agent_progress["logs"].append("Building agent graph...")
        await asyncio.sleep(1)
        
        agent_progress["logs"].append("Running analysis...")
        await asyncio.sleep(2)
        
        agent_progress["status"] = "completed"
        agent_progress["logs"].append("AI analysis completed!")
        
        return {"status": "success", "insights": ["Sample insight 1", "Sample insight 2"]}
        
    except Exception as e:
        agent_progress["status"] = "error"
        agent_progress["error"] = str(e)
        agent_progress["logs"].append(f"Error: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
