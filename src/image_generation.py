from typing import Optional
from openai import AsyncOpenAI

from src.constants import OPENAI_API_BASE, IMAGE_GENERATION_MODEL
from src.utils import logger

client = AsyncOpenAI(base_url=OPENAI_API_BASE)

async def generate_image(prompt: str) -> Optional[str]:
    """Generate an image URL for the given prompt."""
    try:
        response = await client.images.generate(
            prompt=prompt,
            n=1,
            model=IMAGE_GENERATION_MODEL,
        )
        if response.data:
            return response.data[0].url
    except Exception as e:
        logger.exception(e)
    return None
