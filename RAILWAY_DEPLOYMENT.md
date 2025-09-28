# Railway Deployment Guide

## 🚂 Deploying Your Thompson Sampling Dashboard on Railway

Railway is perfect for your FastAPI app with heavy dependencies like pandas, numpy, and scipy. Unlike Vercel, Railway has no size limits and excellent Python support.

## Prerequisites
- A Railway account (sign up at https://railway.app)
- Your OpenAI API key
- Git repository (GitHub, GitLab, or Bitbucket)

## Step 1: Prepare Your Repository

Your app is already configured for Railway deployment with:
- ✅ `Procfile` - Tells Railway how to start your app
- ✅ `railway.json` - Railway configuration
- ✅ `requirements.txt` - Python dependencies
- ✅ Environment variable setup for OpenAI API key

## Step 2: Deploy to Railway

### Option A: Deploy via Railway Dashboard (Recommended)

1. **Push your code to Git:**
   ```bash
   git add .
   git commit -m "Add Railway deployment configuration"
   git push origin main
   ```

2. **Connect to Railway:**
   - Go to https://railway.app/dashboard
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway will auto-detect it's a Python app

3. **Configure Environment Variables:**
   - In your Railway project dashboard
   - Go to "Variables" tab
   - Add: `OPENAI_API_KEY` = your actual OpenAI API key
   - Optionally add: `OPENAI_MODEL` = "gpt-4o-mini" (or your preferred model)

4. **Deploy:**
   - Railway will automatically build and deploy
   - Your app will be available at `https://your-project-name.railway.app`

### Option B: Deploy via Railway CLI

1. **Install Railway CLI:**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login to Railway:**
   ```bash
   railway login
   ```

3. **Deploy:**
   ```bash
   cd ad-agent
   railway init
   railway up
   ```

4. **Set Environment Variables:**
   ```bash
   railway variables set OPENAI_API_KEY=your_actual_api_key_here
   ```

## Step 3: Test Your Deployment

1. Visit your Railway URL
2. Test the Thompson Sampling functionality
3. Verify the AI agents are working with your OpenAI API key

## Railway vs Vercel Comparison

| Feature | Railway | Vercel |
|---------|---------|---------|
| Python Support | ✅ Excellent | ⚠️ Limited |
| Size Limits | ✅ No limits | ❌ 250MB limit |
| Dependencies | ✅ Full pip/conda support | ⚠️ Limited |
| Background Tasks | ✅ Supported | ❌ Not supported |
| Database | ✅ Built-in PostgreSQL | ❌ External only |
| Storage | ✅ Persistent | ❌ Temporary only |
| Pricing | ✅ Generous free tier | ⚠️ Limited free tier |

## Environment Variables Reference

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes | - |
| `OPENAI_MODEL` | OpenAI model to use | No | gpt-4o-mini |
| `PORT` | Port for the app | No | Railway sets this |

## File Structure for Railway

```
ad-agent/
├── app.py              # Main FastAPI app
├── Procfile            # Railway start command
├── railway.json        # Railway configuration
├── requirements.txt    # Python dependencies
├── agents/            # AI agents module
├── bandit/            # Thompson sampling module
├── sim/               # Simulation module
└── data/              # Data files
```

## Troubleshooting

### Common Issues:

1. **Build Failures:**
   - Check that all dependencies are in `requirements.txt`
   - Ensure Python version compatibility (Railway uses Python 3.11 by default)

2. **Environment Variables Not Working:**
   - Verify the variable name is exactly `OPENAI_API_KEY`
   - Redeploy after adding environment variables

3. **Import Errors:**
   - Check that all local modules are properly structured
   - Ensure relative imports are correct

4. **App Not Starting:**
   - Check the Railway logs in the dashboard
   - Verify the Procfile command is correct

## Railway Dashboard Features

- **Logs**: Real-time application logs
- **Metrics**: CPU, memory, and network usage
- **Variables**: Environment variable management
- **Deployments**: Deployment history and rollback
- **Domains**: Custom domain management

## Next Steps

After successful deployment:
1. Set up a custom domain (optional)
2. Configure automatic deployments from your Git repository
3. Monitor usage and performance in Railway dashboard
4. Set up monitoring and alerts if needed
5. Consider upgrading to Railway Pro for production use

## Cost

- **Free Tier**: $5 credit monthly (usually enough for development)
- **Pro**: $20/month for production use
- **Pay-as-you-go**: Only pay for what you use

Railway is much more cost-effective than Vercel for Python applications with heavy dependencies!
