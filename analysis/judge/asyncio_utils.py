import asyncio
import json
from typing import Type, TypeAlias, Literal
from tqdm.asyncio import tqdm_asyncio


Message: TypeAlias = dict[Literal["role", "content"], str]
Messages: TypeAlias = list[Message]


def async_retry_with_exponential_backoff(
    max_retries: int = 20,
    initial_sleep_time: float = 1.0,
    backoff_factor: float = 1.5,
):
    """
    Async retry decorator with exponential backoff for rate limit errors.

    Args:
        max_retries (int): Maximum number of retry attempts. Defaults to 20.
        initial_sleep_time (float): Initial sleep time in seconds. Defaults to 1.
        backoff_factor (float): Factor by which the sleep time increases after each retry. Defaults to 1.5.

    Returns:
        callable: A decorator that wraps async functions with retry logic.

    Raises:
        Exception: If the maximum number of retries is exceeded.
        Any other exception raised by the function that is not a rate limit error.
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            sleep_time = initial_sleep_time

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if "rate limit" in str(e).lower().replace("_", " "):
                        if attempt < max_retries - 1:
                            await asyncio.sleep(sleep_time)
                            sleep_time *= backoff_factor
                        else:
                            raise Exception(f"Maximum retries {max_retries} exceeded")
                    else:
                        raise e

            raise Exception(f"Maximum retries {max_retries} exceeded")

        return wrapper

    return decorator


@async_retry_with_exponential_backoff()
async def generate_structured_response(
    client,
    model: str,
    messages: Messages,
    response_format: Type,
    temperature: float = 1,
    max_completion_tokens: int = 1000,
) -> dict:
    """Generate a response using the AsyncOpenAI API with structured output."""
    try:
        response = await client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            response_format=response_format,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in generation: {e}. Returning empty data.")
        empty_data = {field: "" for field in response_format.__annotations__.keys()}
        return empty_data


async def generate_structured_responses_async(
    client,
    model: str,
    messages_list: list[Messages],
    response_format: Type,
    temperature: float = 1,
    max_completion_tokens: int = 1000,
    max_concurrent: int = 20,
    show_progress: bool = True,
) -> list[dict]:
    """
    Generate multiple structured responses using asyncio.

    Args:
        client: AsyncOpenAI client instance
        model: Model name to use
        messages_list: List of message lists to process
        response_format: Pydantic model for structured output
        temperature: Sampling temperature (default: 1)
        max_completion_tokens: Maximum tokens in response (default: 1000)
        max_concurrent: Maximum concurrent requests (default: 20)
        show_progress: Show progress bar (default: True)

    Returns:
        List of structured response dictionaries in the same order as input
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_with_semaphore(messages):
        async with semaphore:
            return await generate_structured_response(
                client=client,
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )

    tasks = [generate_with_semaphore(msg) for msg in messages_list]

    if show_progress:
        # Use tqdm.asyncio for async progress bar
        results = await tqdm_asyncio.gather(*tasks, desc="Processing")
    else:
        results = await asyncio.gather(*tasks)

    return results


@async_retry_with_exponential_backoff()
async def generate_response(
    client,
    model: str,
    messages: Messages,
    temperature: float = 1,
    max_completion_tokens: int = 1000,
    top_p: float | None = None,
    seed: int | None = None,
    n: int = 1,
):
    """Generate a response using the AsyncOpenAI API without structured output."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            seed=seed,
            n=n,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in generation: {e}. Returning None.")
        return None


async def generate_responses_async(
    client,
    model: str,
    messages_list: list[Messages],
    temperature: float = 1,
    max_completion_tokens: int = 1000,
    top_p: float | None = None,
    seed: int | None = None,
    n: int = 1,
    max_concurrent: int = 32,
    show_progress: bool = True,
    desc: str | None = None,
) -> list:
    """
    Generate multiple responses using asyncio without structured output.

    Args:
        client: AsyncOpenAI client instance
        model: Model name to use
        messages_list: List of message lists to process
        temperature: Sampling temperature (default: 1)
        max_completion_tokens: Maximum tokens in response (default: 1000)
        top_p: Nucleus sampling parameter (optional)
        seed: Random seed for deterministic output (optional)
        n: Number of completions per request (default: 1)
        max_concurrent: Maximum concurrent requests (default: 32)
        show_progress: Show progress bar (default: True)

    Returns:
        List of response objects in the same order as input
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_with_semaphore(messages):
        async with semaphore:
            return await generate_response(
                client=client,
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                top_p=top_p,
                seed=seed,
                n=n,
            )

    tasks = [generate_with_semaphore(msg) for msg in messages_list]

    if show_progress:
        # Use tqdm.asyncio for async progress bar
        results = await tqdm_asyncio.gather(*tasks, desc=desc)
    else:
        results = await asyncio.gather(*tasks)

    return results
