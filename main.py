#!/usr/bin/env python3
"""
Taber — Stock & ETF Research Agent

Usage:
    python main.py "Analyze the SPDR S&P 500 ETF (SPY)"
    python main.py --query "Compare MSFT and GOOGL"
    python main.py --interactive
"""

import argparse
import sys

from dotenv import load_dotenv

# Must load .env before agent imports so API keys are available at instantiation
load_dotenv()

from agent import build_agent, run_agent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="taber",
        description="Stock and ETF research agent powered by Claude + Tavily",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("query", nargs="?", help="Research query (positional)")
    group.add_argument("--query", "-q", dest="query_flag", metavar="QUERY",
                       help="Research query (named flag)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run in interactive REPL mode")
    parser.add_argument("--stream", "-s", action="store_true", default=True,
                        help="Stream output as generated (default: True)")
    parser.add_argument("--no-stream", action="store_false", dest="stream",
                        help="Wait for full response before printing")
    parser.add_argument("--model", "-m", default="claude-sonnet-4-6",
                        help="Claude model to use (default: claude-sonnet-4-6)")
    return parser.parse_args()


def run_interactive(agent, stream: bool) -> None:
    print("Taber Research Agent — type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            query = input("Research query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break
        print("\n--- Research Results ---")
        result = run_agent(query, agent=agent, stream=stream)
        if not stream:
            print(result)
        print("\n" + "─" * 60 + "\n")


def main() -> None:
    args = parse_args()
    query = args.query or getattr(args, "query_flag", None)

    if not query and not args.interactive:
        print(
            "Error: provide a query as a positional argument, "
            "with --query/-q, or use --interactive/-i for REPL mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        agent = build_agent(model_name=args.model)
    except EnvironmentError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.interactive:
        run_interactive(agent, stream=args.stream)
    else:
        try:
            result = run_agent(query, agent=agent, stream=args.stream)
            if not args.stream:
                print(result)
        except Exception as e:
            print(f"Agent error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
