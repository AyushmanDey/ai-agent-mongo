import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import ALL_TOOLS

load_dotenv()

SYSTEM_PROMPT = """You are a helpful assistant with access to a personal knowledge base.

How to use your tools:
- For specific questions, call search_knowledge.
- To summarize a whole document, first call list_documents to find the exact title,
  then call get_document.
- If the user asks what you know about or what documents exist, call list_documents.

Rules:
- Base your answers on tool results, and mention which document the answer came from.
- If the tools return nothing relevant, say you don't know. Do not make things up.
- For greetings or small talk, just reply normally without tools."""


def get_llm_with_tools():
    llm = ChatGroq(model=os.getenv("GROQ_MODEL"), temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("messages"),   # the conversation goes here
    ])

    # prompt -> LLM that knows about our tools
    return prompt | llm.bind_tools(ALL_TOOLS)