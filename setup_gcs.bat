@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: setup_gcs.bat — Create GCS bucket and upload 3D model placeholders
:: Usage: setup_gcs.bat YOUR_PROJECT_ID
:: ─────────────────────────────────────────────────────────────────────────────

SET PROJECT_ID=%1
IF "%PROJECT_ID%"=="" (
    echo ERROR: Provide your project ID.
    echo Usage: setup_gcs.bat YOUR_PROJECT_ID
    exit /b 1
)

SET BUCKET=ai-3d-models-bucket

echo.
echo ══════════════════════════════════════════════════
echo   GCS Bucket Setup — %BUCKET%
echo ══════════════════════════════════════════════════

:: Create bucket (ignore error if already exists)
echo [1/3] Creating bucket gs://%BUCKET%...
gsutil mb -l us-central1 gs://%BUCKET%

:: Set CORS policy so Unity WebRequest can access models
echo [2/3] Setting CORS policy...
echo [{"origin":["*"],"method":["GET"],"maxAgeSeconds":3600}] > cors.json
gsutil cors set cors.json gs://%BUCKET%
del cors.json

:: Make bucket publicly readable
echo [3/3] Setting public access...
gsutil iam ch allUsers:objectViewer gs://%BUCKET%

echo.
echo ══════════════════════════════════════════════════
echo   BUCKET READY: https://storage.googleapis.com/%BUCKET%/
echo.
echo   Now upload your .glb models:
echo   gsutil cp *.glb gs://%BUCKET%/models/
echo ══════════════════════════════════════════════════
