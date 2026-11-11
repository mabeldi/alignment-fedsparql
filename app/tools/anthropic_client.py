from dotenv import load_dotenv
load_dotenv()
import os 
from anthropic import Anthropic, AsyncAnthropic
# Anthropic names the fields input_tokens/output_tokens; record() reads both spellings.
from tools.token_usage import record

client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
clientAsync = AsyncAnthropic(api_key=os.getenv("CLAUDE_API_KEY"))

def anthropic_execute(messages: str, temperature=0.2 , model="claude-opus-4-20250514", system: str = None, json_mode=False):
    print("anthropic_execute")
    system_message = system
    filtered_messages = []


    for msg in messages:
        if msg["role"] == "system":
            system_message = msg["content"]
        else:
            filtered_messages.append(msg)
    response = client.messages.create(
        model=model,
        messages= filtered_messages,
        max_tokens=4096,
        system=system_message,
        temperature=temperature
    )
    record("anthropic", model, getattr(response, "usage", None))
    return response.content[0].text.strip()



async def anthropic_async_execute(messages: str, temperature=0.2, model="claude-opus-4-20250514", json_mode=False):
    print("anthropic_async_execute")
    system_message = None
    filtered_messages = []

    for msg in messages:
        if msg["role"] == "system":
            system_message = msg["content"]
        else:
            filtered_messages.append(msg)

    response = await clientAsync.messages.create(
            model=model,
            messages=filtered_messages,
            max_tokens=4096,
            system=system_message,
            temperature=temperature
        )
    record("anthropic", model, getattr(response, "usage", None))
    return response.content[0].text.strip()