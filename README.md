# SAM2 Backend - Beam.cloud Deployment

## Quick Deployment

### Setup Instructions

#### 1. Get your Beam token
- Log into [Beam.cloud](https://beam.cloud)
- Go to **Settings → API Keys**
- Copy your authentication token

#### 2. Add GitHub Secret
- Go to your GitHub repository
- Navigate to **Settings → Secrets and variables → Actions**
- Click **New repository secret**
  - **Name:** `BEAM_TOKEN`
  - **Value:** Your Beam.cloud authentication token

#### 3. Deploy
- Push any changes to the `main` branch
- GitHub Actions will automatically deploy to Beam.cloud
- Monitor deployment in the **Actions** tab

---

## Manual Deployment (Optional)

```bash
# Install Beam CLI
pip install beam-client

# Configure authentication
beam configure default --token YOUR_BEAM_TOKEN

# Deploy
beam deploy app.py:handler
```

---

That's it! Your SAM2 backend will be live on Beam.cloud with automatic