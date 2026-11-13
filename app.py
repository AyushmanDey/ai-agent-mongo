import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.mongodb import MongoDBSaver

from agent import build_graph
from tools import list_documents
import ingest

load_dotenv()
st.set_page_config(page_title="Knowledge Agent", page_icon="🤖", layout="centered")


# ---------- Build the agent once (not on every click) ----------
@st.cache_resource
def get_app():
    client = MongoClient(os.getenv("MONGODB_URI"))
    checkpointer = MongoDBSaver(client, db_name="checkpointing_db")
    return build_graph(checkpointer=checkpointer)


app = get_app()


# ---------- Sidebar ----------
with st.sidebar:
    st.title("🤖 Knowledge Agent")
    st.caption("MongoDB · LangGraph · Groq")

    thread_id = st.text_input(
        "Conversation name",
        value="default",
        help="Same name = continue an old chat (memory is stored in MongoDB).",
    )

    st.divider()
    st.subheader("📚 Documents")
    st.text(list_documents.invoke({}))

    uploaded = st.file_uploader(
        "Add documents", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if st.button("Upload & rebuild knowledge base", disabled=not uploaded):
        Path("knowledge").mkdir(exist_ok=True)
        for f in uploaded:
            (Path("knowledge") / f.name).write_bytes(f.getbuffer())
        with st.spinner("Chunking, embedding and indexing..."):
            ingest.main()
        st.success("Knowledge base updated!")
        st.rerun()


config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}


# ---------- Helper: show the tool calls the agent made ----------
def show_steps(messages):
    calls = [c for m in messages if isinstance(m, AIMessage) for c in (m.tool_calls or [])]
    if calls:
        with st.expander(f"🔧 Agent steps ({len(calls)} tool call{'s' if len(calls) > 1 else ''})"):
            for c in calls:
                st.code(f"{c['name']}({c['args']})", language="python")


# ---------- Show previous messages from MongoDB memory ----------
history = app.get_state(config).values.get("messages", [])
for m in history:
    if isinstance(m, HumanMessage):
        st.chat_message("user").write(m.content)
    elif isinstance(m, AIMessage) and m.content and not m.tool_calls:
        st.chat_message("assistant").write(m.content)

if not history:
    st.info("Ask me anything about your documents. Try: *What is the BrightPath refund policy?*")


# ---------- Chat input ----------
if question := st.chat_input("Ask a question..."):
    st.chat_message("user").write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            before = len(history)
            result = app.invoke({"messages": [("user", question)]}, config)
            new_messages = result["messages"][before:]

        show_steps(new_messages)
        st.write(result["messages"][-1].content)