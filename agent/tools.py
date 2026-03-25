"""
Tool definitions for the stock/ETF research agent.
"""

import os

import modal
from langchain_core.tools import tool
from langchain_tavily import TavilySearch


def get_sandbox_tool():
    """
    Build and return the Modal sandbox execution tool.

    Raises:
        EnvironmentError: If MODAL_TOKEN_ID or MODAL_TOKEN_SECRET is not set.
    """
    if not os.environ.get("MODAL_TOKEN_ID") or not os.environ.get(
        "MODAL_TOKEN_SECRET"
    ):
        raise EnvironmentError(
            "MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are not set. "
            "Copy .env.example to .env and add your Modal credentials, "
            "or run: modal token new"
        )

    @tool
    def run_in_sandbox(command: str) -> str:
        """
        Execute a shell command in a secure Modal cloud sandbox and return its output.
        Use this to run Python scripts, perform calculations, process data, or any
        task that requires running code. The sandbox has internet access and a
        standard Python environment. Returns stdout and stderr combined.
        """
        sb = modal.Sandbox.create()
        try:
            proc = sb.exec("bash", "-c", command)
            stdout = proc.stdout.read()
            stderr = proc.stderr.read()
            returncode = proc.wait()
            parts = []
            if stdout:
                parts.append(f"stdout:\n{stdout}")
            if stderr:
                parts.append(f"stderr:\n{stderr}")
            if returncode != 0:
                parts.append(f"return code: {returncode}")
            return "\n".join(parts) if parts else "(no output)"
        finally:
            sb.terminate()

    return run_in_sandbox


def get_search_tool(max_results: int = 5) -> TavilySearch:
    """
    Build and return the Tavily search tool.

    Raises:
        EnvironmentError: If TAVILY_API_KEY is not set.
    """
    if not os.environ.get("TAVILY_API_KEY"):
        raise EnvironmentError(
            "TAVILY_API_KEY is not set. "
            "Copy .env.example to .env and add your Tavily API key."
        )
    return TavilySearch(
        max_results=max_results,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        name="tavily_search",
        description=(
            "Search the web for current information about stocks, ETFs, "
            "market data, financial news, SEC filings, analyst ratings, "
            "and economic indicators. Use specific ticker symbols or "
            "company names for best results."
        ),
    )


def get_tools(max_results: int = 5) -> list:
    """Return the full list of tools available to the agent."""
    return [get_search_tool(max_results=max_results), get_sandbox_tool()]
