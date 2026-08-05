import argparse

from lib import hybrid_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Accepts a list of scores and prints normalized values")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="The list of scores to be normalized")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            hybrid_search.normalize(args.scores)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
