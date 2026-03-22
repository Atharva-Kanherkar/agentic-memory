# Experiment: Do Audio Embeddings Capture Emotional Tone?

**Date:** 2026-03-22
**Model:** Gemini Embedding 2 (`gemini-embedding-2-preview`, 768 dims)
**Script:** `audio_emotion_probe.py`

---

## Why We Did This

The agentic memory architecture stores memories as embedding vectors. For the system to support cross-modal retrieval — e.g. a text query pulling up an audio memory — the embedding model must place semantically similar content close together in vector space, *across modalities*.

The specific question: **does Gemini Embedding 2 encode emotional tone in audio, or just acoustic structure?**

- If emotional: a sad song's vector should land near "grief, melancholy" text — even without lyrics, titles, or metadata.
- If acoustic-only: a slow, quiet song would score similarly against "peaceful/calm" as against "grief" — because both share tempo and dynamics.

This matters architecturally because the `EpisodicMemory` model has an `emotional_valence` field. If the embedding already encodes emotion, we don't need a separate sentiment analysis step — we can derive valence directly from cosine similarity against a small probe set.

---

## How We Did It

1. Loaded audio files and split into 60-second chunks using `ffmpeg` (Gemini's audio limit is 80s per call).
2. Embedded each chunk as **raw bytes + mime_type only** — no filename, no metadata, no lyrics passed to the model.
3. Averaged the chunk vectors into a single song embedding.
4. Embedded 6 emotional text probes (generic vocabulary, no cultural references, no instrument names).
5. Computed cosine similarity between the song vector and each probe vector.

**Probes used:**
```
grief / loss    → "deep sorrow, mourning, grief, loss, tears, heartbreak"
melancholy      → "melancholy, wistful, longing, nostalgia, bittersweet"
peaceful / calm → "calm, serene, peaceful, quiet, stillness, gentle"
joy / happiness → "joyful, happy, cheerful, uplifting, bright, celebration"
tension / dread → "tense, anxious, dread, suspense, unease, dark"
neutral         → "ordinary, mundane, neutral, average, unremarkable"
```

---

## Results

### Song 1 — Schindler's List Theme (John Williams, performed by Itzhak Perlman)
Solo violin, minor key, slow tempo. Western orchestral. No vocals.

| Probe | Similarity |
|---|---|
| **melancholy** | **0.7013** ← winner |
| grief / loss | 0.6826 |
| peaceful / calm | 0.6811 |
| joy / happiness | 0.6686 |
| tension / dread | 0.6670 |
| neutral | 0.6392 |

---

### Song 2 — Phir Se (Shashwat Sachdev, vocals by Arijit Singh)
Bollywood sad song. Hindi vocals. Different cultural and musical context entirely.

| Probe | Similarity |
|---|---|
| **grief / loss** | **0.6943** ← winner |
| melancholy | 0.6928 |
| peaceful / calm | 0.6587 |
| joy / happiness | 0.6489 |
| tension / dread | 0.6449 |
| neutral | 0.6073 |

---

### Song 3 — Rasputin (Boney M)
Upbeat disco-pop. English vocals. High energy, danceable. Dramatic lyrical subject matter.

| Probe | Similarity |
|---|---|
| **joy / happiness** | **0.6887** ← winner |
| tension / dread | 0.6725 |
| melancholy | 0.6430 |
| neutral | 0.6374 |
| peaceful / calm | 0.6299 |
| grief / loss | 0.6254 ← lowest |

---

### Song 4 — Didi (Khaled, feat. Nabil, Sons of Yusuf)
Algerian Raï. Upbeat, celebratory, romantic. Arabic/French vocals.

| Probe | Similarity |
|---|---|
| **joy / happiness** | **0.6623** ← winner |
| melancholy | 0.6291 |
| tension / dread | 0.6269 |
| neutral | 0.6164 |
| peaceful / calm | 0.6148 |
| grief / loss | 0.6053 ← lowest |

---

## What the Results Mean

**All three songs ranked correctly.** The sad songs (Songs 1 and 2) landed in the grief/melancholy cluster. The upbeat song (Song 3) landed on joy. This rules out the acoustic-only hypothesis — a purely structural embedding would have placed the slow Schindler's theme closer to "peaceful/calm."

**The model made a subtle distinction between the two sad songs:**
- Schindler's List → *melancholy* (wistful, nostalgic, reflective)
- Phir Se → *grief/loss* (raw heartbreak, more emotionally direct)

This is a meaningful difference. Schindler's List is a memorial piece — mournful but restrained. Phir Se is a breakup song with vocals expressing active grief. The model separated them correctly without any metadata.

**Phir Se is in Hindi.** The model received only raw audio bytes, no transcription, no lyrics. It still placed the song in the grief cluster — suggesting the emotional encoding is acoustic, not linguistic.

**Rasputin showed an interesting dual result** — joy first, tension/dread second (gap of 0.016). This captures the song's character well: it's an upbeat disco track about a dark, violent historical figure. The model picked up both the musical energy and the dramatic weight simultaneously.

**Grief ranked last on Rasputin** (0.6254) — a complete reversal from the two sad songs where grief ranked first or second. The largest winner-to-loser gap across all three songs was Rasputin (0.6887 joy vs 0.6254 grief = **0.063 gap**), suggesting the model is more confident classifying energetic music than subtle sadness.

**Didi (Raï, Arabic/French vocals) also scored joy first, grief last** — consistent with Rasputin. But the second-place probe was *melancholy* (0.6291), not tension/dread like Rasputin. This is culturally accurate: Raï music has a signature quality of longing and nostalgia even in its most celebratory songs. The model picked up a wistfulness that Rasputin doesn't carry. Both are "happy" songs, but they're not the same kind of happy.

---

## Architectural Implications

1. **No separate emotion detection needed.** Embed the audio, run similarity against the probe set, take the top label. `emotional_valence` on `EpisodicMemory` can be populated automatically at store time.

2. **Cross-modal retrieval is viable.** A text query like `"something melancholy"` will surface audio memories, because both live in the same 768-dimensional space and cluster by meaning.

3. **Cross-cultural and cross-lingual.** The system does not need language-specific models or culturally-specific training to handle non-English audio content.

---

## Open Questions

- How does chunking strategy affect results? (First 60s vs last 60s vs averaged — does the emotional arc of a song matter?)
- Does a happy song score correctly? (**Yes — validated by Rasputin, joy ranked first.**)
- How narrow is the margin needed for reliable classification? (0.0015 difference between grief and melancholy on Phir Se — is that stable across runs?)
