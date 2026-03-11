"""
download_models.py
Called during Docker build to pre-download and cache ML model weights.
This makes container startup instant (no runtime downloads).
"""
import sys

print("=" * 50)
print("Pre-downloading ML model weights into image...")
print("=" * 50)

# 1. SentenceTransformer (all-MiniLM-L6-v2 ~ 90MB)
try:
    from sentence_transformers import SentenceTransformer
    print("Downloading SentenceTransformer: all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Verify it works
    _ = model.encode(["test embedding"], normalize_embeddings=True)
    print("SentenceTransformer weights ready.")
except Exception as e:
    print(f"WARNING: SentenceTransformer download failed: {e}", file=sys.stderr)

# 2. OpenCLIP (ViT-B-32 ~ 350MB)
try:
    import open_clip
    print("Downloading CLIP: ViT-B-32 (openai) ...")
    model_clip, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    print("CLIP weights ready.")
except Exception as e:
    print(f"WARNING: CLIP download failed (will fallback at runtime): {e}", file=sys.stderr)

print("=" * 50)
print("Model pre-download complete.")
print("=" * 50)
