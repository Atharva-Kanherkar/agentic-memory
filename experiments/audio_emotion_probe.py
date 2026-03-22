"""
Experiment: Do audio embeddings capture emotional tone?

We chunk an audio file into 60s segments, embed each chunk as raw bytes
(no filename, no metadata sent to the model), average the vectors, then
compute cosine similarity against a set of emotional text probes.

If the model captures emotional tone:
  sad probes should score higher than happy probes.
If the model captures acoustic structure only:
  scores will be roughly uniform across emotional categories.
"""

import sys
import os
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.embeddings import GeminiEmbedder

MEDIA_DIR = os.path.join(os.path.dirname(__file__), "../media")
CHUNK_SECONDS = 60   # 60 seconds — within Gemini's 80s limit
MIME_TYPE = "audio/mpeg"

# ── emotional probes ────────────────────────────────────────────────────────
# Deliberately generic — no song titles, no cultural references.
PROBES = {
    "grief / loss":       "deep sorrow, mourning, grief, loss, tears, heartbreak",
    "melancholy":         "melancholy, wistful, longing, nostalgia, bittersweet",
    "peaceful / calm":    "calm, serene, peaceful, quiet, stillness, gentle",
    "joy / happiness":    "joyful, happy, cheerful, uplifting, bright, celebration",
    "tension / dread":    "tense, anxious, dread, suspense, unease, dark",
    "neutral":            "ordinary, mundane, neutral, average, unremarkable",
}

# ── helpers ─────────────────────────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def get_duration_seconds(file_path: str) -> float:
    """Ask ffprobe for the audio duration without loading the file into memory."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", file_path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def chunk_audio(file_path: str, chunk_seconds: int) -> list[bytes]:
    """Split audio into fixed-length chunks using ffmpeg. Returns raw MP3 bytes per chunk."""
    total = get_duration_seconds(file_path)
    chunks = []

    start = 0
    while start < total:
        result = subprocess.run(
            ["ffmpeg", "-ss", str(start), "-t", str(chunk_seconds),
             "-i", file_path, "-f", "mp3", "-q:a", "4", "pipe:1"],
            capture_output=True, check=True,
        )
        chunks.append(result.stdout)
        start += chunk_seconds

    return chunks


def average_embeddings(vectors: list[list[float]]) -> list[float]:
    """Element-wise mean across all chunk vectors."""
    arr = np.array(vectors)
    return arr.mean(axis=0).tolist()


# ── main ─────────────────────────────────────────────────────────────────────

def run():
    if len(sys.argv) < 2:
        files = [f for f in os.listdir(MEDIA_DIR) if f.endswith(".mp3")]
        print("Usage: python audio_emotion_probe.py <filename>")
        print(f"Files in media/: {files}")
        sys.exit(1)

    audio_file = os.path.join(MEDIA_DIR, sys.argv[1])
    if not os.path.exists(audio_file):
        print(f"File not found: {audio_file}")
        sys.exit(1)

    embedder = GeminiEmbedder()

    print(f"Loading and chunking audio...")
    chunks = chunk_audio(audio_file, CHUNK_SECONDS)
    print(f"  {len(chunks)} chunks of up to {CHUNK_SECONDS}s each")

    print(f"Embedding chunks (raw bytes only — no filename sent to model)...")
    chunk_vectors = []
    for i, chunk_bytes in enumerate(chunks):
        vec = embedder.embed_audio(chunk_bytes, mime_type=MIME_TYPE)
        chunk_vectors.append(vec)
        print(f"  chunk {i+1}/{len(chunks)} done ({len(vec)} dims)")

    audio_vec = average_embeddings(chunk_vectors)
    print(f"\nAudio embedding ready ({len(audio_vec)} dims, averaged over {len(chunks)} chunks)")

    print(f"\nEmbedding emotional probes...")
    probe_vectors = {label: embedder.embed_text(text) for label, text in PROBES.items()}

    print(f"\n{'─' * 45}")
    print(f"{'Emotional probe':<25}  {'Similarity':>10}")
    print(f"{'─' * 45}")

    scores = {label: cosine_similarity(audio_vec, vec) for label, vec in probe_vectors.items()}
    for label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 80)
        print(f"{label:<25}  {score:>10.4f}  {bar}")

    print(f"{'─' * 45}")
    winner = max(scores, key=scores.get)
    print(f"\nClosest match: {winner}")


if __name__ == "__main__":
    run()
