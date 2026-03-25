"""
Modal deployment for the Tabor stock/ETF research agent.

Deploy:
    tabor deploy

Run remotely after deploying:
    modal run modal_app.py --query "Analyze SPY"
"""

import modal

app = modal.App("tabor")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "langchain>=1.2.10",
        "langchain-anthropic>=1.3.4",
        "langchain-tavily>=0.2.17",
        "langsmith>=0.2.0",
    )
    .copy_local_dir("agent", "/root/agent")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv()],
)
def research(query: str, model: str = "claude-sonnet-4-6") -> str:
    """Run a stock/ETF research query and return the analysis."""
    import sys

    sys.path.insert(0, "/root")
    from agent import build_agent, run_agent

    agent = build_agent(model_name=model)
    return run_agent(query, agent=agent, stream=False)


@app.local_entrypoint()
def main(query: str, model: str = "claude-sonnet-4-6"):
    """Local entrypoint: invoke the remote research function."""
    result = research.remote(query=query, model=model)
    print(result)
