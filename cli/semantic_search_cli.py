import argparse

from lib import semantic_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Confirm that the local LLM is loaded and functional")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embeds text using the configured model")
    embed_text_parser.add_argument("text", type=str, help="The text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify embedding generation from movies dataset")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embed a particular query string")
    embed_query_parser.add_argument("query", type=str, help="The query to be embedded")

    search_parser = subparsers.add_parser("search", help="Search through the dataset with a specific query")
    search_parser.add_argument("query", type=str, help="The query to be embedded")
    search_parser.add_argument("--limit", type=int, default=5, help="The number of results to show")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk provided text")
    chunk_parser.add_argument("text", type=str, help="The text to be chunked")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="The number of words to include in each chunk")
    chunk_parser.add_argument("--overlap", type=int, default=40, help="The number of words that should be included as overlap in each chunk")

    chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk provided text")
    chunk_parser.add_argument("text", type=str, help="The text to be chunked")
    chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="The number of sentences to include in each chunk")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="The number of sentences that should be included as overlap in each chunk")


    args = parser.parse_args()
    match args.command:
        case "verify":
            semantic_search.verify_model()
        case "embed_text":
            semantic_search.embed_text(args.text)
        case "verify_embeddings":
            semantic_search.verify_embeddings()
        case "embed_query":
            semantic_search.embed_query_text(args.query)
        case "search":
            semantic_search.search(args.query, args.limit)
        case "chunk":
            semantic_search.chunk(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_search.semantic_chunk(args.text, args.max_chunk_size, args.overlap)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
