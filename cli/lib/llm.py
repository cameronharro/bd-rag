import os
from dotenv import load_dotenv
from openai import OpenAI

def get_llm_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    return client

def enhancement_query(query: str, method: str) -> str:
    match method:
        case "spell":
            return f"""Fix any spelling errors in the user-provided movie search query below.
                Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                Preserve punctuation and capitalization unless a change is required for a typo fix.
                If there are no spelling errors, or if you're unsure, output the original query unchanged.
                Output only the final query text with trimmed whitespace, nothing else.
                User query: "{query}"
                """
        case _:
            raise ValueError(f"invalid enhancement method: {method}")

