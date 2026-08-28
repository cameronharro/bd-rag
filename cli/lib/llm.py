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
        case "rewrite":
            return f"""Rewrite the user-provided movie search query below to be more specific and searchable.

            Consider:
            - Common movie knowledge (famous actors, popular films)
            - Genre conventions (horror = scary, animation = cartoon)
            - Keep the rewritten query concise (under 10 words)
            - It should be a Google-style search query, specific enough to yield relevant results
            - Don't use boolean logic

            Examples:
            - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
            - "movie about bear in london with marmalade" -> "Paddington London marmalade"
            - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

            If you cannot improve the query, output the original unchanged.
            Output only the rewritten query text, nothing else.

            User query: "{query}"
            """
        case _:
            raise ValueError(f"invalid enhancement method: {method}")

