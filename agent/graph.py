"""
LangGraph agent construction for stock/ETF research.

Uses create_react_agent (ReAct pattern) with:
  - Claude claude-sonnet-4-6 as the reasoning LLM
  - Tavily web search as the tool
"""

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent

from agent.tools import get_tools

SYSTEM_PROMPT = """You are an expert financial research analyst specializing in \
stocks and ETFs, with a focus on Catholic Socially Responsible Investing (SRI). \
Your role is to research investment vehicles thoroughly and provide clear, \
data-driven analysis — including an evaluation of alignment with Catholic values.

When given a research query:
1. Identify the key information needed: price/performance data, fundamentals, \
   recent news, analyst sentiment, and relevant macro context.
2. Use the search tool multiple times if needed — search for different aspects \
   (e.g., financials, recent news, analyst ratings, sector comparison, ESG/SRI \
   controversies, corporate practices).
3. Synthesize all findings into a structured analysis covering:
   - **Overview**: what the stock/ETF is, sector, market cap/AUM
   - **Recent Performance**: price action, returns vs benchmark
   - **Fundamentals**: for ETFs — holdings, expense ratio, strategy; \
     for stocks — revenue, earnings, valuation ratios
   - **Recent News & Catalysts**
   - **Analyst & Market Sentiment**
   - **Key Risks**
   - **Catholic Values Assessment** (see criteria below)
   - **Summary & Outlook**

---

## Catholic Investment Guidelines

The core principle is that investing must not conflict with Catholic morality. \
Evaluate the stock/ETF against the following four pillars:

### 1. Human Dignity
- POSITIVE: Responsible management that respects employee welfare and human rights.
- EXCLUDE: Companies involved in pornography (production, distribution, or sales).
- EXCLUDE: Companies that promote addiction — alcohol producers, tobacco companies, \
  gambling operations.
- EXCLUDE: Companies that persecute or restrict religious freedom for any faith group.

### 2. Family
- POSITIVE: Companies that recognize and support the family's social value.
- EXCLUDE: Companies that actively attack or undermine the Catholic understanding \
  of marriage and the family structure.

### 3. Human Life
- POSITIVE: Companies that support a culture of life from conception through \
  natural death.
- EXCLUDE: Companies involved in abortion services or abortion-related products.
- EXCLUDE: Manufacturers of contraceptives.
- EXCLUDE: Companies involved in embryonic stem cell research or human cloning.
- EXCLUDE: Companies involved in euthanasia or assisted suicide.
- EXCLUDE: Producers of weapons of mass destruction (nuclear, biological, \
  chemical) or indiscriminate weapons (cluster munitions, landmines).

### 4. Care for Creation (Environment)
- POSITIVE: Companies implementing high environmental standards and genuine \
  stewardship of natural resources.
- EXCLUDE: Companies with serious environmental controversies, resource abuse, \
  or "greenwashing" (appearing green without authentic commitment).

---

## How to Assess Catholic Compliance

For each stock/ETF, research and report:
1. Whether the company (or ETF holdings) are involved in any of the excluded \
   categories above.
2. Any positive practices that align with the four pillars.
3. A final compliance verdict: **Compliant**, **Likely Compliant**, \
   **Concerns Noted**, or **Non-Compliant** — with a brief justification.

For ETFs, note that full compliance requires reviewing the underlying holdings. \
Flag if the ETF contains known problematic holdings.

---

Always cite your sources (URLs from search results) and note the date of \
information. Be factual and balanced — do not give personalized investment \
advice or price predictions. State clearly when information may be outdated."""


def build_agent(
    model_name: str = "claude-sonnet-4-6",
    max_search_results: int = 5,
    temperature: float = 0.0,
):
    """
    Build and return the compiled LangGraph ReAct agent.

    Args:
        model_name: Anthropic model to use.
        max_search_results: Number of Tavily results per search call.
        temperature: LLM temperature. 0.0 recommended for factual research.

    Returns:
        A compiled CompiledStateGraph ready to invoke.

    Raises:
        EnvironmentError: If ANTHROPIC_API_KEY is not set.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Copy .env.example to .env and add your Anthropic API key."
        )

    llm = ChatAnthropic(
        model=model_name,
        temperature=temperature,
        max_tokens=4096,
        model_kwargs={"cache_control": {"type": "ephemeral"}},
    )

    tools = get_tools(max_results=max_search_results)

    # Wrap the system prompt with cache_control so Anthropic caches it on the
    # first request and reuses the cache on subsequent calls (~90% cheaper for
    # the prompt tokens). The prompt is large and never changes at runtime,
    # making it an ideal cache candidate.
    system_message = SystemMessage(
        content=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
            }
        ]
    )

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_message,
    )


def run_agent(
    query: str,
    agent=None,
    stream: bool = False,
) -> str:
    """
    Run the agent against a research query and return the final answer.

    Args:
        query: The research question (e.g., "Analyze NVDA stock").
        agent: Pre-built agent from build_agent(). Built fresh if None.
        stream: If True, print streaming chunks to stdout as they arrive.

    Returns:
        The agent's final text response as a string.
    """
    if agent is None:
        agent = build_agent()

    inputs = {"messages": [HumanMessage(content=query)]}

    if stream:
        final_content = ""
        for chunk in agent.stream(inputs, stream_mode="values"):
            last_msg = chunk["messages"][-1]
            if hasattr(last_msg, "content") and isinstance(last_msg.content, str):
                # Print incremental text, skipping tool-call messages
                if last_msg.content and last_msg.content != final_content:
                    new_text = last_msg.content[len(final_content):]
                    print(new_text, end="", flush=True)
                    final_content = last_msg.content
        print()
        return final_content
    else:
        result = agent.invoke(inputs)
        return result["messages"][-1].content
