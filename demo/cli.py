import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.semantic import SemanticMemory
from stores.semantic_store import SemanticStore


def cmd_store(args):
    store = SemanticStore()
    record = SemanticMemory(content=args.content)
    record_id = store.store(record)
    print(f"Stored [{record_id[:8]}]: {args.content}")


def main():
    parser = argparse.ArgumentParser(description="Agentic Memory CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    store_p = sub.add_parser("store", help="Store a new memory")
    store_p.add_argument("content", type=str, help="Text content to store")

    args = parser.parse_args()

    if args.command == "store":
        cmd_store(args)


if __name__ == "__main__":
    main()
