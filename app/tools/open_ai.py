import os
from dotenv import load_dotenv
from openai import OpenAI
from openai import AsyncOpenAI

from tools.token_usage import record
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

gpt_client = AsyncOpenAI()

def _extra(json_mode):
    return {"response_format": {"type": "json_object"}} if json_mode else {}


def open_ai_execute(messages: str, temperature=0.2, model="gpt-4o", json_mode=False):
    response = client.chat.completions.create(
        model=model,
        messages= messages,
        temperature=temperature,
        **_extra(json_mode)
    )
    record("openai", model, getattr(response, "usage", None))
    return response.choices[0].message.content.strip()


async def open_ai_async_execute(messages: str, temperature=0.2, model="gpt-4o", json_mode=False):
    response = await gpt_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **_extra(json_mode)
        )
    record("openai", model, getattr(response, "usage", None))
    return response.choices[0].message.content.strip()