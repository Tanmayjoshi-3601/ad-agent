# Vercel Deployment Guide

## Prerequisites
1. A Vercel account (sign up at https://vercel.com)
2. Your OpenAI API key
3. Git repository (GitHub, GitLab, or Bitbucket)

## Step 1: Prepare Your Repository

Your app is already configured for Vercel deployment with:
- ✅ `vercel.json` configuration file
- ✅ Clean `requirements.txt` with pinned versions
- ✅ Environment variable setup for OpenAI API key

## Step 2: Deploy to Vercel

### Option A: Deploy via Vercel Dashboard (Recommended)

1. **Push your code to Git:**
   ```bash
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push origin main
   ```

2. **Connect to Vercel:**
   - Go to https://vercel.com/dashboard
   - Click "New Project"
   - Import your Git repository
   - Vercel will auto-detect it's a Python/FastAPI app

3. **Configure Environment Variables:**
   - In the Vercel dashboard, go to your project settings
   - Navigate to "Environment Variables"
   - Add: `OPENAI_API_KEY` with your actual API key
   - Optionally add: `OPENAI_MODEL` (defaults to "gpt-4o-mini")

4. **Deploy:**
   - Click "Deploy" - Vercel will build and deploy automatically
   - Your app will be available at `https://your-project-name.vercel.app`

### Option B: Deploy via Vercel CLI

1. **Install Vercel CLI:**
   ```bash
   npm i -g vercel
   ```

2. **Login to Vercel:**
   ```bash
   vercel login
   ```

3. **Deploy:**
   ```bash
   cd ad-agent
   vercel
   ```

4. **Set Environment Variables:**
   ```bash
   vercel env add OPENAI_API_KEY
   # Enter your API key when prompted
   ```

5. **Redeploy with environment variables:**
   ```bash
   vercel --prod
   ```

## Step 3: Test Your Deployment

1. Visit your deployed URL
2. Test the Thompson Sampling functionality
3. Verify the AI agents are working with your OpenAI API key

## Troubleshooting

### Common Issues:

1. **Build Failures:**
   - Check that all dependencies are in `requirements.txt`
   - Ensure Python version compatibility

2. **Environment Variables Not Working:**
   - Verify the variable name is exactly `OPENAI_API_KEY`
   - Redeploy after adding environment variables

3. **Import Errors:**
   - Check that all local modules are properly structured
   - Ensure relative imports are correct

### File Structure for Vercel:
```
ad-agent/
├── app.py              # Main FastAPI app
├── vercel.json         # Vercel configuration
├── requirements.txt    # Python dependencies
├── agents/            # AI agents module
├── bandit/            # Thompson sampling module
├── sim/               # Simulation module
└── data/              # Data files
```

## Environment Variables Reference

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes | - |
| `OPENAI_MODEL` | OpenAI model to use | No | gpt-4o-mini |

## Next Steps

After successful deployment:
1. Set up a custom domain (optional)
2. Configure automatic deployments from your Git repository
3. Monitor usage and performance in Vercel dashboard
4. Set up monitoring and alerts if needed
