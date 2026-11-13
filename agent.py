from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import ToolMessage
from langchain_core.messages import ToolMessage, trim_messages
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from llm import get_llm_with_tools
from tools import ALL_TOOLS

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


# 1. STATE: what flows through the graph.
#    add_messages means new messages are APPENDED, not overwritten.
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(checkpointer=None):
    llm_with_tools = get_llm_with_tools()

    # 2. AGENT NODE: the LLM looks at all messages and decides what to do.
    def agent_node(state: GraphState):
        # Send only the most recent messages to the LLM.
        # Full history stays saved in MongoDB.
        recent = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=len,     # count messages, not tokens (simple)
            max_tokens=20,         # keep the last ~20 messages
            start_on="human",      # never start mid tool-call
        )
        response = llm_with_tools.invoke({"messages": recent})
        return {"messages": [response]}
    # 3. TOOL NODE: run every tool the LLM asked for.
    def tool_node(state: GraphState):
        last_message = state["messages"][-1]
        results = []
        for call in last_message.tool_calls:
            print(f"   🔧 {call['name']}({call['args']})")
            tool = TOOLS_BY_NAME[call["name"]]
            output = tool.invoke(call["args"])
            results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        return {"messages": results}

    # 4. ROUTER: after the agent speaks, go to tools or finish?
    def route(state: GraphState):
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    # 5. WIRE IT UP
    graph = StateGraph(GraphState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")   # after tools, always back to the agent

    return graph.compile(checkpointer=checkpointer)


def ask(app, question: str, thread_id: str = "1"):
    print(f"\n🧑 {question}")
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}
    result = app.invoke({"messages": [("user", question)]}, config)
    print(f"🤖 {result['messages'][-1].content}")


if __name__ == "__main__":
    app = build_graph()
    ask(app, "What is the refund policy at BrightPath?")
    ask(app, "Summarize the strength training document")
    ask(app, "My bench press has been stuck for weeks. What should I do?")
    ask(app, "What is the capital of France?")