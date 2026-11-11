from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
from tools.token_usage import record
load_dotenv()
client = InferenceClient(token=os.getenv("HUGFACE_API_KEY"))
import asyncio

def llma3_execute(messages: str, temperature=0.2, model="meta-llama/Meta-Llama-3-70B-Instruct", json_mode=False):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    record("huggingface", model, getattr(completion, "usage", None))
    return completion.choices[0].message.content




def _chat_sync(messages, temperature=0.2, model="meta-llama/Meta-Llama-3-70B-Instruct"):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    record("huggingface", model, getattr(completion, "usage", None))
    return completion.choices[0].message.content

async def llma3_async_execute(messages, temperature=0.2, model="meta-llama/Meta-Llama-3-70B-Instruct", json_mode=False):
    return await asyncio.to_thread(_chat_sync, messages, temperature, model)
    




# completion = client.chat.completions.create(
#     model="meta-llama/Meta-Llama-3-8B-Instruct",
#     messages=[
#         {"role": "system", "content": "You are a SPARQL-query generating assistant."},
#         {"role": "user", "content": "Generate a SPARQL query to retrieve the names of authors who wrote more than one paper."}
#     ],
#     max_tokens=150,
#     temperature=0.0,
# )

# print(completion.choices[0].message.content)

