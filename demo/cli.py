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


def cmd_query(args):
    store = SemanticStore()
    results = store.retrieve(args.query, top_k=args.top_k)
    if not results:
        print("No results found.")
        return
    for rank, (record, score) in enumerate(results, 1):
        print(f"  {rank}. [{score:.4f}] {record.content}")


def main():
    parser = argparse.ArgumentParser(description="Agentic Memory CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    store_p = sub.add_parser("store", help="Store a new memory")
    store_p.add_argument("content", type=str, help="Text content to store")

    query_p = sub.add_parser("query", help="Search memories by meaning")
    query_p.add_argument("query", type=str, help="Natural language query")
    query_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    if args.command == "store":
        cmd_store(args)
    elif args.command == "query":
        cmd_query(args)


if __name__ == "__main__":
    main()
