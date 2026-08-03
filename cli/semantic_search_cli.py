import argparse

from lib import semantic_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Confirm that the local LLM is loaded and functional")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embeds text using the configured model")
    embed_text_parser.add_argument("text", type=str, help="The text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify embedding generation from movies dataset")

    args = parser.parse_args()
    match args.command:
        case "verify":
            semantic_search.verify_model()
        case "embed_text":
            semantic_search.embed_text(args.text)
        case "verify_embeddings":
            semantic_search.verify_embeddings()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
