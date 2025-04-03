import json
import aiohttp
import asyncio
from aiohttp import ClientTimeout


# Create a timeout object (values in seconds)
timeout = ClientTimeout(total=420)


def get_prompt(content):
    prompt = [{"role": "user", "content": content}]
    return prompt


async def predict(
    session,
    prompt,
    model=None,
    adaptor=None,
    inference_url="http://localhost:8338/v1/chat/completions",
    api_key="dummy",
    max_tokens=1024,
    temperature=0.01,
    top_p=1.0,
):
    # If prompt is already a list (i.e. a conversation), use it as the messages payload.
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = [{"role": "user", "content": prompt}]
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = json.dumps(
        {
            "model": model,
            "adaptor": adaptor,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
    )

    async with session.post(inference_url, headers=headers, data=payload, timeout=timeout) as response:
        try:
            # response_obj = await response
            response_json = await response.json()
            # return response_json["choices"][0]["message"]["content"]
            return response_json
        except Exception as e:
            print(f"Exception: {e}")
            print(await response.text())
            return ""


async def process_batch(
    session,
    batch,
    model=None,
    adaptor=None,
    inference_url="http://localhost:8338/v1/chat/completions",
    api_key="dummy",
    temperature=0.01,
    max_tokens=1024,
    top_p=1.0,
    max_concurrent=1,  # Limit concurrent requests
):
    results = []
    # Process in smaller groups with limited concurrency
    for i in range(0, len(batch), max_concurrent):
        sub_batch = batch[i : i + max_concurrent]
        tasks = [
            predict(
                session,
                conversation,
                model=model,
                adaptor=adaptor,
                inference_url=inference_url,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            for conversation in sub_batch
        ]
        sub_results = await asyncio.gather(*tasks)
        results.extend(sub_results)

        # Add a small delay between batches to allow MLX operations to complete
        await asyncio.sleep(0.01)

    return results


async def process_dataset(
    examples,
    batch_size,
    model=None,
    adaptor=None,
    inference_url="http://localhost:8338/v1/chat/completions",
    api_key="dummy",
    temperature=0.01,
    max_tokens=1024,
    top_p=1.0,
    min_idx=0,
    max_idx=None,
):
    """
    Process a list of conversations that each contain messages.
    Returns an array of assistant responses arranged based on the global index.
    """
    print(f"Processing {len(examples)} examples")
    if max_idx is None:
        max_idx = len(examples)
    START_FROM_SAMPLE_INDEX = min_idx
    END_AT_SAMPLE_INDEX = max_idx
    if isinstance(batch_size, str):
        batch_size = int(batch_size)

    responses = {}  # dictionary to hold responses with global_idx as key
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for start in range(min_idx, max_idx, batch_size):
            end = min(start + batch_size, max_idx)
            batch = examples[start:end]
            results = await process_batch(
                session,
                batch,
                model=model,
                adaptor=adaptor,
                inference_url=inference_url,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            for idx, assistant_response in enumerate(results):
                global_idx = start + idx
                if START_FROM_SAMPLE_INDEX <= global_idx < END_AT_SAMPLE_INDEX:
                    responses[global_idx] = assistant_response

    # Arrange responses in order of their global index
    ordered_responses = [responses[i] for i in sorted(responses.keys())]
    return ordered_responses


# async def process_dataset(
#     examples,
#     batch_size,
#     model=None,
#     adaptor=None,
#     inference_url="http://localhost:8338/v1/chat/completions",
#     api_key="dummy",
#     temperature=0.01,
#     max_tokens=1024,
#     top_p=1.0,
#     min_idx=0,
#     max_idx=None,
# ):
#     """
#     Process a JSON list of conversations instead of a DataFrame.
#     Each conversation in `examples` is a list of message dictionaries.
#     For each conversation, the assistant response is obtained and appended.
#     """
#     print(f"Processing {len(examples)} examples")
#     if max_idx is None:
#         max_idx = len(examples)
#     START_FROM_SAMPLE_INDEX = min_idx
#     END_AT_SAMPLE_INDEX = max_idx
#     if isinstance(batch_size, str):
#         batch_size = int(batch_size)
#     responses = {} # dictionary to hold
#     async with aiohttp.ClientSession(timeout=timeout) as session:
#         for start in range(min_idx, max_idx, batch_size):
#             end = min(start + batch_size, max_idx)
#             batch = examples[start:end]  # using list slicing
#             print(f"Processing batch {start}:{end}")
#             results = await process_batch(
#                 session,
#                 batch,
#                 model=model,
#                 adaptor=adaptor,
#                 inference_url=inference_url,
#                 api_key=api_key,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 top_p=top_p,
#             )
#             # For each conversation, append the assistant's response as a new message
#             for idx, assistant_response in enumerate(results):
#                 global_idx = start + idx
#                 if START_FROM_SAMPLE_INDEX <= global_idx < END_AT_SAMPLE_INDEX:
#                     # Append the result to the conversation
#                     examples[global_idx].append({"role": "assistant", "content": assistant_response})
#     return examples
