"""Streamlit web UI for the Gnani.ai bot. All logic lives in rag.py.

Run:  streamlit run app.py
"""
from collections import deque

import streamlit as st

from rag import HISTORY_TURNS, MIN_SIM, RagBot

st.set_page_config(page_title="Gnani.ai Assistant", page_icon="🎙️", layout="centered")

EXAMPLES = [
    "What does Gnani.ai do?",
    "Which languages does Gnani.ai support?",
    "Which industries does Gnani.ai serve?",
    "Tell me about Gnani's speech recognition technology",
]


@st.cache_resource(show_spinner="Loading models...")
def get_bot() -> RagBot:
    return RagBot()          # embedding model + vector DB are loaded once and reused


try:
    bot = get_bot()
except Exception as e:       # missing .env, missing chroma_db, etc.
    st.error(f"Could not start the bot: {e}")
    st.info("Check your .env (OPENROUTER_API_KEY, OPENROUTER_MODEL) and run `python ingest.py` first.")
    st.stop()

# per-visitor state
st.session_state.setdefault("messages", [])   # what is shown on screen
st.session_state.setdefault("pairs", [])      # (question, answer) memory for follow-ups
st.session_state.setdefault("pending", None)

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Gnani.ai Assistant")
    st.caption("Answers come only from the gnani.ai website content. "
               "Off-topic questions and prompt-injection attempts are blocked.")
    st.subheader("Try asking")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    st.divider()
    show_debug = st.toggle("Show debug info")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pairs = []
        st.rerun()


def render(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for url in msg["sources"]:
                    st.markdown(f"- {url}")
        if show_debug and msg.get("status"):
            st.caption(f"status: `{msg['status']}` · min similarity: {MIN_SIM}")
            for h in msg.get("hits", []):
                st.caption(f"{h['sim']:.2f}  {h['url']}")


# ---------------- main chat ----------------
st.title("🎙️ Gnani.ai Assistant")

if not st.session_state.messages:
    st.markdown("Ask me about Gnani.ai's products, voice AI, languages and use cases.")

for m in st.session_state.messages:
    render(m)

prompt = st.chat_input("Ask about Gnani.ai...")
if not prompt and st.session_state.pending:
    prompt, st.session_state.pending = st.session_state.pending, None

if prompt:
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    render(user_msg)

    # the bot is shared across visitors, so load this visitor's history before asking
    bot.history = deque(st.session_state.pairs, maxlen=HISTORY_TURNS)
    with st.spinner("Thinking..."):
        r = bot.ask(prompt)
    if r.status == "ok":
        st.session_state.pairs = list(bot.history)

    reply = {"role": "assistant", "content": r.answer, "sources": r.sources[:3],
             "status": r.status, "hits": r.hits}
    st.session_state.messages.append(reply)
    render(reply)