from google import genai
from google.genai import types

from config import GEMINI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS


class GeminiEmbedder:
    """Converts content (text, image, audio) into embedding vectors using Gemini Embedding 2."""

    def __init__(self):
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._config = types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS)

    def _embed(self, contents: list) -> list[list[float]]:
        """Core call — returns one vector per item in contents."""
        result = self._client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=contents,
            config=self._config,
        )
        return [e.values for e in result.embeddings]

    def embed_text(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def embed_image(self, image_bytes: bytes, mime_type: str = "image/png") -> list[float]:
        part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        return self._embed([part])[0]

    def embed_audio(self, audio_bytes: bytes, mime_type: str = "audio/mpeg") -> list[float]:
        part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
        return self._embed([part])[0]
