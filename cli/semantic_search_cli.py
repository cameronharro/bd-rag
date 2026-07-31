import argparse

from lib import semantic_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Confirm that the local LLM is loaded and functional")

    args = parser.parse_args()
    match args.command:
        case "verify":
            semantic_search.verify_model()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
