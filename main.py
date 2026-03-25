#!/usr/bin/env python3
"""
Tabor — Stock & ETF Research Agent

Usage:
    tabor run "Analyze the SPDR S&P 500 ETF (SPY)"
    tabor run --interactive
    tabor deploy
"""

import subprocess
import sys

import click
from dotenv import load_dotenv

# Must load .env before agent imports so API keys are available at instantiation
load_dotenv()

from agent import build_agent, run_agent  # noqa: E402

BANNER = """\
╔══════════════════════════════════════════╗
║        Tabor Research Agent              ║
║  Catholic SRI · Stocks · ETFs            ║
╚══════════════════════════════════════════╝"""

DIVIDER = "─" * 60


def _run_query(agent, query: str, stream: bool) -> None:
    """Execute a single query and print results."""
    try:
        result = run_agent(query, agent=agent, stream=stream)
        if not stream:
            click.echo(result)
    except Exception as e:
        click.secho(f"Agent error: {e}", fg="red", err=True)
        sys.exit(1)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version="0.1.0", prog_name="tabor")
def cli():
    """Tabor — Stock & ETF research powered by Claude + Tavily."""


@cli.command()
@click.argument("query", required=False)
@click.option("--interactive", "-i", is_flag=True,
              help="Run in interactive REPL mode.")
@click.option("--stream/--no-stream", "-s/-S", default=True, show_default=True,
              help="Stream output as it is generated.")
@click.option("--model", "-m", default="claude-sonnet-4-6", show_default=True,
              metavar="MODEL", help="Claude model to use.")
def run(query: str | None, interactive: bool, stream: bool, model: str) -> None:
    """Run a research query against the Tabor agent.

    Provide a QUERY directly, or pass --interactive / -i for a REPL session.

    \b
    Examples:
      tabor run "Analyze SPY"
      tabor run --interactive
      tabor run -i --no-stream
    """
    if not query and not interactive:
        raise click.UsageError(
            "Provide a QUERY argument or use --interactive / -i for REPL mode."
        )

    try:
        agent = build_agent(model_name=model)
    except EnvironmentError as e:
        click.secho(f"Configuration error: {e}", fg="red", err=True)
        sys.exit(1)

    if interactive:
        click.secho(BANNER, fg="cyan", bold=True)
        click.echo()
        click.secho("Type 'exit', 'quit', or Ctrl-C to stop.\n", dim=True)

        while True:
            try:
                query_input = click.prompt(
                    click.style("Research query", fg="green", bold=True),
                    prompt_suffix=" › ",
                )
            except (click.Abort, EOFError):
                click.echo()
                click.secho("Goodbye.", fg="yellow")
                break

            query_input = query_input.strip()
            if not query_input:
                continue
            if query_input.lower() in {"exit", "quit", "q"}:
                click.secho("Goodbye.", fg="yellow")
                break

            click.echo()
            click.secho("Research Results", fg="cyan", bold=True)
            click.secho(DIVIDER, dim=True)
            _run_query(agent, query_input, stream)
            click.secho(DIVIDER + "\n", dim=True)
    else:
        _run_query(agent, query, stream)


@cli.command()
def deploy() -> None:
    """Deploy the Tabor agent to Modal.

    Requires the Modal CLI to be installed and authenticated.
    API keys are read from your local .env file via modal.Secret.from_dotenv().

    \b
    Prerequisites:
      pip install modal
      modal setup

    \b
    Example:
      tabor deploy
    """
    click.secho("Deploying Tabor agent to Modal...", fg="cyan", bold=True)
    result = subprocess.run(
        ["modal", "deploy", "modal_app.py"],
        check=False,
    )
    sys.exit(result.returncode)


# Keep `python main.py` working
main = cli

if __name__ == "__main__":
    cli()
