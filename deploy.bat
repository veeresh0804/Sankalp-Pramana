@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: deploy.bat — One-click Google Cloud Run deployment
:: Usage: deploy.bat YOUR_PROJECT_ID [REGION]
:: Example: deploy.bat my-gcp-project us-central1
:: ─────────────────────────────────────────────────────────────────────────────

SET PROJECT_ID=%1
SET REGION=%2

IF "%PROJECT_ID%"=="" (
    echo ERROR: Please provide your Google Cloud Project ID.
    echo Usage: deploy.bat YOUR_PROJECT_ID [REGION]
    echo Example: deploy.bat my-gcp-project us-central1
    exit /b 1
)

IF "%REGION%"=="" SET REGION=us-central1

SET SERVICE_NAME=ai-3d-backend
SET IMAGE=gcr.io/%PROJECT_ID%/%SERVICE_NAME%

echo.
echo ══════════════════════════════════════════════════════
echo   AI 3D Backend — Cloud Run Deployment
echo   Project : %PROJECT_ID%
echo   Region  : %REGION%
echo   Image   : %IMAGE%
echo ══════════════════════════════════════════════════════
echo.

:: Step 1 — Set active GCP project
echo [1/5] Setting GCP project...
gcloud config set project %PROJECT_ID%
IF ERRORLEVEL 1 goto :error

:: Step 2 — Enable required APIs
echo [2/5] Enabling Cloud APIs (run, build, storage)...
gcloud services enable run.googleapis.com cloudbuild.googleapis.com containerregistry.googleapis.com storage.googleapis.com
IF ERRORLEVEL 1 goto :error

:: Step 3 — Build and push container image
echo [3/5] Building container image with Cloud Build...
gcloud builds submit --tag %IMAGE% .
IF ERRORLEVEL 1 goto :error

:: Step 4 — Deploy to Cloud Run
echo [4/5] Deploying to Cloud Run...
gcloud run deploy %SERVICE_NAME% ^
    --image %IMAGE% ^
    --platform managed ^
    --region %REGION% ^
    --allow-unauthenticated ^
    --memory 2Gi ^
    --cpu 2 ^
    --min-instances 1 ^
    --max-instances 10 ^
    --timeout 120 ^
    --set-env-vars "GCS_BUCKET_NAME=ai-3d-models-bucket,GCS_PROJECT_ID=%PROJECT_ID%,RETRIEVAL_SOURCES=sketchfab,polyhaven,mock"
:: NOTE: Manage SKETCHFAB_API_TOKEN and GEMINI_API_KEY via Cloud Run Console or Secret Manager
IF ERRORLEVEL 1 goto :error

:: Step 5 — Print the live URL
echo [5/5] Fetching deployed URL...
FOR /F "tokens=*" %%i IN ('gcloud run services describe %SERVICE_NAME% --region %REGION% --format=value(status.url)') DO SET SERVICE_URL=%%i

echo.
echo ══════════════════════════════════════════════════════
echo   DEPLOYMENT SUCCESSFUL
echo   API URL : %SERVICE_URL%
echo   Docs    : %SERVICE_URL%/docs
echo   Health  : %SERVICE_URL%/health
echo ══════════════════════════════════════════════════════
echo.

goto :end

:error
echo.
echo ══════════════════════════════════════════════════════
echo   DEPLOYMENT FAILED — see error above
echo ══════════════════════════════════════════════════════
exit /b 1

:end
