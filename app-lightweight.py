"""
Lightweight version of the app for Vercel deployment
This version removes heavy dependencies to stay under the 250MB limit
"""

import json
import os
from datetime import datetime, timedelta
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Dict, List

app = FastAPI(title="Thompson Sampling Dashboard - Lightweight")

# Global state for progress tracking
ts_progress = {"status": "idle", "current_day": 0, "total_days": 0, "logs": [], "error": None}
agent_progress = {"status": "idle", "step": "", "insights": [], "suggestions": {}, "error": None}

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Thompson Sampling Dashboard - Lightweight</title>
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
                max-width: 1200px;
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
            
            .info-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin-top: 1rem;
            }
            
            .info-item {
                background: var(--bg-tertiary);
                padding: 1rem;
                border-radius: 8px;
                border: 1px solid var(--border-secondary);
            }
            
            .info-label {
                color: var(--text-muted);
                font-size: 0.9rem;
                margin-bottom: 0.25rem;
            }
            
            .info-value {
                color: var(--text-primary);
                font-weight: 600;
            }
            
            .note {
                background: rgba(88, 166, 255, 0.1);
                border: 1px solid rgba(88, 166, 255, 0.3);
                border-radius: 8px;
                padding: 1rem;
                margin-top: 2rem;
            }
            
            .note h3 {
                color: var(--accent-primary);
                margin-bottom: 0.5rem;
            }
            
            .note p {
                color: var(--text-secondary);
                margin-bottom: 0.5rem;
            }
            
            .note ul {
                color: var(--text-secondary);
                margin-left: 1.5rem;
            }
            
            .note li {
                margin-bottom: 0.25rem;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Thompson Sampling Dashboard</h1>
                <p>AI-Powered Ad Budget Optimization</p>
            </div>
            
            <div class="status-card">
                <h2>System Status</h2>
                <div class="status idle">Lightweight Mode</div>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Deployment</div>
                        <div class="info-value">Vercel</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Mode</div>
                        <div class="info-value">Lightweight</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Status</div>
                        <div class="info-value">Ready</div>
                    </div>
                </div>
            </div>
            
            <div class="note">
                <h3>🚀 Lightweight Deployment</h3>
                <p>This is a lightweight version of the Thompson Sampling Dashboard optimized for Vercel's serverless environment.</p>
                <p><strong>Features available:</strong></p>
                <ul>
                    <li>✅ FastAPI backend</li>
                    <li>✅ OpenAI integration</li>
                    <li>✅ Basic API endpoints</li>
                    <li>✅ Environment variable support</li>
                </ul>
                <p><strong>For full functionality:</strong></p>
                <ul>
                    <li>🔧 Use local development with all dependencies</li>
                    <li>🔧 Deploy to a VPS or container service</li>
                    <li>🔧 Use Vercel Pro for larger function limits</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/api/status")
def get_status():
    """Get current system status"""
    return {
        "status": "running",
        "mode": "lightweight",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0-lightweight"
    }

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/test-openai")
def test_openai():
    """Test OpenAI API connection"""
    try:
        from openai import OpenAI
        client = OpenAI()
        
        # Simple test call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            max_tokens=10
        )
        
        return {
            "status": "success",
            "message": "OpenAI API is working",
            "response": response.choices[0].message.content
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"OpenAI API error: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
