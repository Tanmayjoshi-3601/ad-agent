
# 🧠 Smart Budget Optimization Dashboard (The app is live here: grand-fulfillment-production.up.railway.app)

An AI-powered marketing budget optimization system that uses Thompson Sampling and AI agents to automatically learn and optimize campaign budget allocation across customer segments.

## 📖 Project Overview

This system demonstrates how artificial intelligence can optimize marketing spend by:

1. **Learning from Performance Data**: Uses Thompson Sampling to understand which customer segments perform best
2. **Automated Budget Allocation**: Dynamically adjusts budget distribution based on real performance
3. **AI-Powered Insights**: Generates business insights and recommendations using Large Language Models

The dashboard simulates 7 days of marketing campaigns, showing how the system learns and improves budget allocation over time.

## 🔬 The Mathematics Behind Thompson Sampling

### Core Concept
Thompson Sampling is a **Bayesian bandit algorithm** that solves the exploration-exploitation trade-off in budget allocation.

### Mathematical Foundation

**1. Beta-Binomial Model**
- Each customer segment is modeled as a Beta distribution: `Beta(α, β)`
- **α (alpha)**: Number of successful conversions + prior successes
- **β (beta)**: Number of failed attempts + prior failures
- **Conversion Rate θ**: `θ ~ Beta(α, β)`

**2. Thompson Sampling Process**
```
For each day:
1. Sample θ̃ₖ ~ Beta(αₖ, βₖ) for each segment k
2. Allocate budget proportional to sampled θ̃ₖ values
3. Observe clicks and conversions
4. Update: αₖ ← αₖ + conversions, βₖ ← βₖ + (clicks - conversions)
```

**3. Budget Allocation**
```
share_k = θ̃ₖ / Σⱼ θ̃ⱼ  (probability matching)
dollars_k = share_k × total_budget
```

**4. Pacing Constraints**
- **Floor constraint**: `share_k ≥ floor` (minimum budget per segment)
- **Cap constraint**: `|share_k(t) - share_k(t-1)| ≤ cap` (max daily change)

### Why This Works
- **Exploration**: Uncertain segments get higher variance in sampling → more exploration
- **Exploitation**: High-performing segments get higher mean sampling → more budget
- **Automatic Learning**: As data accumulates, uncertainty decreases and allocation stabilizes

## 🛠️ Local Setup Instructions

### Prerequisites
- Python 3.8+ installed
- OpenAI API key (for AI insights feature)

### Step 1: Clone and Setup
```bash
# If downloading from Replit, extract the files to a local directory
cd your-project-directory

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Environment Configuration
Create a `.env` file in the project root:
```bash
# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

**Important**: Replace `your_openai_api_key_here` with your actual OpenAI API key.

### Step 3: Data Setup
The project includes sample data in `data/facts_daily.parquet`. This contains:
- Customer segment performance data
- Daily clicks, conversions, and spend by segment
- 7 days of historical campaign data

## 🚀 How to Run

### Option 1: Web Dashboard (Recommended)
```bash
python app.py
```
Then open your browser to: `http://localhost:5000`

### Option 2: Command Line Demo
```bash
python run_loop.py
```
This runs the Thompson Sampling optimization and saves results to `plan.csv`.

### Option 3: AI Analysis Only
```bash
python run_agents.py
```
Analyzes existing `plan.csv` and generates insights.

## 📊 Using the Dashboard

### 1. Start Smart Optimization
Click **"Start Smart Optimization"** to begin the Thompson Sampling process:
- Initializes learning models for each customer segment
- Runs 7 days of simulated campaigns
- Shows real-time progress and results
- Updates budget allocation based on performance

### 2. Get AI Insights  
Click **"Get AI Insights"** after optimization completes:
- **Performance Analyst**: Identifies trends, anomalies, and patterns
- **Budget Strategist**: Recommends specific budget changes
- Provides business rationale for each recommendation

### 3. Interpret Results
- **Budget Split Chart**: Current allocation across segments
- **Performance Trends**: Conversion trends over time
- **Success Rates**: Conversion rates by segment
- **AI Insights**: Detailed analysis and recommendations

## 📁 Project Structure

```
├── agents/           # AI analysis agents
│   └── agents.py     # LLM-powered trend analysis and planning
├── bandit/           # Thompson Sampling core
│   ├── state.py      # Beta distribution management
│   ├── ts.py         # Sampling and allocation logic
│   └── update.py     # Posterior updates
├── sim/              # Campaign simulation
│   └── simulate.py   # Realistic campaign outcome simulation
├── data/             # Sample dataset
│   └── facts_daily.parquet  # Historical campaign data
├── app.py            # Web dashboard (FastAPI)
├── run_loop.py       # CLI Thompson Sampling demo
├── rates.py          # CPC/CVR rate preparation
└── requirements.txt  # Python dependencies
```

## 🎯 What This System Demonstrates

### Business Value
1. **Automated Learning**: No manual A/B test setup required
2. **Continuous Optimization**: Adapts to changing performance daily
3. **Risk Management**: Gradual budget shifts prevent large losses
4. **Actionable Insights**: AI explains what's happening and why

### Technical Innovation
1. **Bayesian Learning**: Principled uncertainty quantification
2. **Multi-Armed Bandits**: Optimal exploration-exploitation balance
3. **Constraint Optimization**: Respects business rules and pacing limits
4. **LLM Integration**: Human-readable analysis of complex data patterns

### Use Cases
- **E-commerce**: Optimize ad spend across product categories
- **SaaS**: Allocate budget between customer acquisition channels  
- **Retail**: Balance spend across geographic regions or demographics
- **Finance**: Optimize marketing spend across customer lifetime value segments

## 🔧 Customization

### Modify Business Rules
Edit `CFG` in `run_loop.py`:
```python
CFG = dict(
    budget=500.0,      # Total daily budget
    cap=0.25,          # Max daily change (25%)
    floor=0.03,        # Min budget per segment (3%)
    days=14            # Simulation length
)
```

### Add Your Data
Replace `data/facts_daily.parquet` with your data containing:
- `day`: Date column
- `segment_key`: Customer segment identifier  
- `clicks`: Number of clicks
- `conv`: Number of conversions
- `spent`: Amount spent

### Customize AI Analysis
Modify prompts in `agents/agents.py` to focus on your specific business metrics and KPIs.

## 📈 Expected Results

After running the system, you should see:
- **Learning Curve**: Budget allocation stabilizes as the system learns
- **Performance Improvement**: Higher-converting segments receive more budget
- **Actionable Insights**: Specific recommendations for further optimization
- **Business Impact**: Simulated improvement in overall conversion rates

## 🚨 Troubleshooting

**"cannot access local variable 'np'"**
- Ensure numpy is properly installed: `pip install numpy`

**"OpenAI API Error"**  
- Check your API key in the `.env` file
- Verify your OpenAI account has available credits

**"No module named 'X'"**
- Install missing dependencies: `pip install -r requirements.txt`

**Port 5000 already in use**
- Change the port in `app.py`: `uvicorn.run(app, host="0.0.0.0", port=8000)`

## 📚 Further Reading

- [Thompson Sampling Paper](https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf)
- [Multi-Armed Bandit Problems](https://en.wikipedia.org/wiki/Multi-armed_bandit)
- [Bayesian Statistics Overview](https://en.wikipedia.org/wiki/Bayesian_statistics)

---

**Built with**: Python, FastAPI, Plotly, OpenAI GPT, LangGraph, Pandas, NumPy
