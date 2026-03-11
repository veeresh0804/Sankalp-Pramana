# AI 3D Backend — Cloud Run Deployment Guide

## Prerequisites

Install and authenticate the Google Cloud SDK:

```bash
# Download: https://cloud.google.com/sdk/docs/install
gcloud auth login
gcloud auth application-default login
```

---

## Step 1 — Create GCS Bucket (one time)

```bash
# Windows
setup_gcs.bat YOUR_PROJECT_ID

# Mac/Linux
gsutil mb -l us-central1 gs://ai-3d-models-bucket
gsutil iam ch allUsers:objectViewer gs://ai-3d-models-bucket
```

Then upload your .glb models:

```bash
gsutil cp path/to/your/models/*.glb gs://ai-3d-models-bucket/models/
```

**Free GLB models to start:**
- [Sketchfab Downloadable](https://sketchfab.com/search?downloadable=true)
- [Poly Haven](https://polyhaven.com/models)

---

## Step 2 — Deploy Backend (one command)

```bash
# Windows — run from your ai-3d-backend folder
deploy.bat YOUR_PROJECT_ID

# OR manual commands:
gcloud config set project YOUR_PROJECT_ID
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/ai-3d-backend
gcloud run deploy ai-3d-backend \
  --image gcr.io/YOUR_PROJECT_ID/ai-3d-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

---

## Step 3 — Add Sketchfab Token (optional but recommended)

```bash
gcloud run services update ai-3d-backend \
  --region us-central1 \
  --update-env-vars SKETCHFAB_API_TOKEN=your_token_here
```

Get token free at: sketchfab.com → Settings → Password & API

---

## Step 4 — Test Your Live API

After deploy you get a URL like:
```
https://ai-3d-backend-xxxxx-uc.a.run.app
```

Test it:
```bash
# Health check
curl https://YOUR_URL.run.app/health

# Search for a model
curl -X POST https://YOUR_URL.run.app/search_model \
  -H "Content-Type: application/json" \
  -d "{\"concept\": \"human heart\", \"top_k\": 5}"
```

Or open: `https://YOUR_URL.run.app/docs` for Swagger UI.

---

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `RETRIEVAL_SOURCES` | Comma list: `sketchfab,polyhaven,mock,objaverse` | `sketchfab,polyhaven,mock` |
| `SKETCHFAB_API_TOKEN` | Free token from sketchfab.com | _(none)_ |
| `GCS_BUCKET_NAME` | Your Cloud Storage bucket | `ai-3d-models-bucket` |
| `GCS_PROJECT_ID` | Your GCP project ID | — |
| `MIN_CONFIDENCE` | Minimum score to return (0–1) | `0.3` |
| `TOP_K_CANDIDATES` | Candidates before ranking | `10` |
| `CACHE_TTL` | Response cache TTL in seconds | `3600` |

---

## Update an Existing Deployment

```bash
# Just rebuild and redeploy — same command
deploy.bat YOUR_PROJECT_ID
```

Cloud Run is zero-downtime — new version goes live instantly.
