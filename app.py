"""Streamlit chat interface for the Tabor Catholic SRI Research Agent."""

import json

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tabor · Catholic SRI Research",
    page_icon="✝️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Agent (cached — survives reruns without rebuilding) ───────────────────────
@st.cache_resource(show_spinner=False)
def get_agent(model_name: str):
    from agent import build_agent
    return build_agent(model_name=model_name)


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_text(content) -> str:
    """Return plain text from an AIMessage content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block["text"] if isinstance(block, dict) and block.get("type") == "text"
            else str(block)
            for block in content
        )
    return str(content)


def format_tool_results(content) -> str:
    """Render Tavily search results as readable markdown."""
    if isinstance(content, list):
        # Already a list of result dicts
        items = content
    else:
        try:
            parsed = json.loads(content)
            # Tavily with include_answer returns {"answer": ..., "results": [...]}
            if isinstance(parsed, dict):
                parts = []
                if parsed.get("answer"):
                    parts.append(f"**Summary:** {parsed['answer']}\n")
                items = parsed.get("results", [])
                for r in items[:5]:
                    title = r.get("title", "")
                    url = r.get("url", "")
                    snippet = r.get("content", "")[:400]
                    parts.append(f"**[{title}]({url})**\n{snippet}…")
                return "\n\n---\n\n".join(parts) or content
            if isinstance(parsed, list):
                items = parsed
            else:
                return str(content)[:2000]
        except (json.JSONDecodeError, TypeError):
            # Plain string — truncate for readability
            text = str(content)
            return text[:2000] + ("…" if len(text) > 2000 else "")

    parts = []
    for r in items[:5]:
        if isinstance(r, dict):
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("content", "")[:400]
            parts.append(f"**[{title}]({url})**\n{snippet}…")
        else:
            parts.append(str(r)[:400])
    return "\n\n---\n\n".join(parts) or str(content)[:2000]


def render_search_steps(steps: list[dict]) -> None:
    """Render past search steps as collapsed expanders."""
    for step in steps:
        with st.expander(f"🔍 Searched: *{step['query']}*", expanded=False):
            if step.get("results"):
                st.markdown(step["results"])
            else:
                st.caption("No results stored.")


# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    # Each entry: {"role": "user"|"assistant", "content": str, "steps": list}
    st.session_state.history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✝ Tabor")
    st.caption("Catholic Socially Responsible Investing")
    st.divider()

    model_name = st.selectbox(
        "Claude model",
        options=[
            "claude-sonnet-4-6",
            "claude-opus-4-7",
            "claude-haiku-4-5-20251001",
        ],
        index=0,
    )

    st.divider()

    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.markdown("**About**")
    st.caption(
        "Tabor evaluates stocks and ETFs against four Catholic investment pillars:\n\n"
        "- 🕊 Human Dignity\n"
        "- 👨‍👩‍👧 Family\n"
        "- ❤️ Human Life\n"
        "- 🌿 Care for Creation\n\n"
        "*Not financial advice.*"
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ✝ Tabor Research Agent")
st.caption("Catholic SRI · Stocks & ETFs · Powered by Claude + Tavily")

# ── Render saved chat history ─────────────────────────────────────────────────
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("steps"):
            render_search_steps(msg["steps"])
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask about a stock or ETF (e.g. 'Analyze AAPL')…"):
    # Display user message immediately
    st.session_state.history.append({"role": "user", "content": user_input, "steps": []})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build / retrieve agent
    try:
        agent = get_agent(model_name)
    except EnvironmentError as e:
        st.error(str(e))
        st.stop()

    # ── Stream agent response ─────────────────────────────────────────────────
    steps: list[dict] = []
    full_response = ""
    seen_tc_ids: set[str] = set()
    seen_tool_ids: set[str] = set()

    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        inputs = {"messages": [HumanMessage(content=user_input)]}

        try:
            with st.status("Researching…", expanded=True) as research_status:
                for chunk in agent.stream(inputs, stream_mode="values"):
                    messages = chunk.get("messages", [])

                    for msg in messages:
                        # New tool calls (Claude decided to search)
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                if tc["id"] not in seen_tc_ids:
                                    seen_tc_ids.add(tc["id"])
                                    query = tc["args"].get(
                                        "query",
                                        tc["args"].get("q", str(tc["args"])),
                                    )
                                    st.write(f"🔍 Searching: **{query}**")
                                    steps.append(
                                        {
                                            "type": "search",
                                            "id": tc["id"],
                                            "query": query,
                                            "results": None,
                                        }
                                    )

                        # Tool results arriving
                        if isinstance(msg, ToolMessage):
                            if msg.tool_call_id not in seen_tool_ids:
                                seen_tool_ids.add(msg.tool_call_id)
                                formatted = format_tool_results(msg.content)
                                for step in steps:
                                    if step["id"] == msg.tool_call_id:
                                        step["results"] = formatted
                                        break
                                st.write("✅ Results received")

                    # Stream the final AI text as it grows
                    last_msg = messages[-1] if messages else None
                    if (
                        last_msg is not None
                        and isinstance(last_msg, AIMessage)
                        and not last_msg.tool_calls
                    ):
                        text = extract_text(last_msg.content)
                        if text and text != full_response:
                            full_response = text
                            response_placeholder.markdown(full_response + "▌")

                n = len(steps)
                label = f"Research complete · {n} search{'es' if n != 1 else ''}"
                research_status.update(label=label, state="complete", expanded=False)

        except Exception as e:
            st.error(f"Agent error: {e}")
            full_response = f"*(Error: {e})*"

        response_placeholder.markdown(full_response)

    # Save to history for subsequent reruns
    st.session_state.history.append(
        {
            "role": "assistant",
            "content": full_response,
            "steps": steps,
        }
    )
