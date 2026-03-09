"""
Tool definitions for the stock/ETF research agent.
"""

import os

from langchain_tavily import TavilySearch


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
    return [get_search_tool(max_results=max_results)]
