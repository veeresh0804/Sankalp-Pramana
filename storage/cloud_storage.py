"""
storage/cloud_storage.py
Google Cloud Storage utility for listing and generating signed URLs for 3D models.

Requires:  pip install google-cloud-storage
           GCS_BUCKET_NAME and GCS_PROJECT_ID set in config / env.
"""

from __future__ import annotations
import logging
from typing import List, Optional

from config import GCS_BUCKET_NAME, GCS_PROJECT_ID

logger = logging.getLogger(__name__)

_client = None   # lazy-loaded google.cloud.storage.Client


def _get_client():
    global _client
    if _client is None:
        try:
            from google.cloud import storage
            _client = storage.Client(project=GCS_PROJECT_ID)
            logger.info("[GCS] Storage client initialised")
        except ImportError:
            logger.warning("[GCS] google-cloud-storage not installed — GCS features disabled")
        except Exception as e:
            logger.error(f"[GCS] Failed to initialise client: {e}")
    return _client


def list_models(prefix: str = "models/") -> List[str]:
    """List all .glb / .obj model URLs in the configured bucket."""
    client = _get_client()
    if client is None:
        return []
    try:
        bucket = client.bucket(GCS_BUCKET_NAME)
        blobs  = bucket.list_blobs(prefix=prefix)
        urls   = [
            f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{b.name}"
            for b in blobs
            if b.name.endswith((".glb", ".obj", ".fbx"))
        ]
        logger.info(f"[GCS] Found {len(urls)} models in gs://{GCS_BUCKET_NAME}/{prefix}")
        return urls
    except Exception as e:
        logger.error(f"[GCS] list_models error: {e}")
        return []


def public_url(blob_name: str) -> str:
    """Return the public HTTPS URL for a blob (bucket must be publicly readable)."""
    return f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{blob_name}"


def signed_url(blob_name: str, expiry_seconds: int = 3600) -> Optional[str]:
    """Generate a time-limited signed URL for a private blob."""
    import datetime
    client = _get_client()
    if client is None:
        return None
    try:
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob   = bucket.blob(blob_name)
        url    = blob.generate_signed_url(
            expiration=datetime.timedelta(seconds=expiry_seconds),
            method="GET",
        )
        return url
    except Exception as e:
        logger.error(f"[GCS] signed_url error: {e}")
        return None
