import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from agent import build_graph

load_dotenv()


def main():
    client = MongoClient(os.getenv("MONGODB_URI"))
    # Saves conversations in the "checkpointing_db" database
    checkpointer = MongoDBSaver(client, db_name="checkpointing_db")

    app = build_graph(checkpointer=checkpointer)

    thread_id = input("Conversation name (same name = continue old chat): ").strip() or "default"
    print(f"\nChatting in thread '{thread_id}'. Type 'exit' to quit.\n")

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}

    while True:
        question = input("🧑 You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        # Only send the NEW message; the checkpointer adds the history.
        result = app.invoke({"messages": [("user", question)]}, config)
        print(f"🤖 Agent: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()