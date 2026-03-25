"""
Tool definitions for the stock/ETF research agent.
"""

import asyncio
import os

from langchain_tavily import TavilySearch
from langchain_core.tools import tool


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


def _run_async(coro):
    """Run an async coroutine from synchronous code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an event loop (e.g. Jupyter); run in a new thread.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


def get_graphlit_tools() -> list:
    """
    Build and return Graphlit knowledge graph tools.

    Returns an empty list if Graphlit credentials are not configured,
    allowing the agent to run without a knowledge graph.
    """
    org_id = os.environ.get("GRAPHLIT_ORGANIZATION_ID")
    env_id = os.environ.get("GRAPHLIT_ENVIRONMENT_ID")
    jwt_secret = os.environ.get("GRAPHLIT_JWT_SECRET")

    if not all([org_id, env_id, jwt_secret]):
        return []

    from graphlit import Graphlit

    client = Graphlit(
        organization_id=org_id,
        environment_id=env_id,
        jwt_secret=jwt_secret,
    )

    @tool
    def graphlit_ingest(title: str, text: str) -> str:
        """
        Ingest text content into the Graphlit knowledge graph for later retrieval.
        Use this to persist research findings, analysis summaries, company profiles,
        or any information worth storing for future queries.

        Args:
            title: A concise, descriptive title for the content (e.g. "NVDA Q4 2024 Analysis").
            text: The full text content to store in the knowledge graph.

        Returns:
            Confirmation message with the assigned content ID, or an error description.
        """
        async def _ingest():
            response = await client.client.ingest_text(
                name=title,
                text=text,
                is_synchronous=True,
            )
            return response.ingest_text.id if response.ingest_text else None

        content_id = _run_async(_ingest())
        if content_id:
            return f"Successfully stored '{title}' in the knowledge graph (ID: {content_id})."
        return f"Failed to store '{title}' in the knowledge graph."

    @tool
    def graphlit_query(query: str) -> str:
        """
        Retrieve relevant content from the Graphlit knowledge graph.
        Use this to look up previously stored research, analysis, or company data
        before performing a new web search, to avoid redundant lookups.

        Args:
            query: A natural-language question or search phrase (e.g. "NVDA Catholic SRI compliance").

        Returns:
            Relevant text excerpts from the knowledge graph, or a message if nothing was found.
        """
        async def _query():
            response = await client.client.retrieve_relevant_contents(
                prompt=query,
            )
            return response.retrieve_relevant_contents

        results = _run_async(_query())

        if not results or not results.results:
            return "No relevant content found in the knowledge graph for that query."

        parts = [f"Found {len(results.results)} relevant result(s) from the knowledge graph:\n"]
        for i, item in enumerate(results.results, 1):
            name = item.content.name if item.content and item.content.name else "Untitled"
            snippet = item.text.strip() if item.text else "(no text)"
            parts.append(f"[{i}] {name}\n{snippet}\n")

        return "\n".join(parts)

    return [graphlit_ingest, graphlit_query]


def get_tools(max_results: int = 5) -> list:
    """Return the full list of tools available to the agent."""
    return [get_search_tool(max_results=max_results), *get_graphlit_tools()]
