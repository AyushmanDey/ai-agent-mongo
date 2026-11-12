from llm import get_llm_with_tools

chain = get_llm_with_tools()

questions = [
    "What is the refund policy at BrightPath?",
    "Summarize the strength training document",
    "What documents do you have?",
    "Hi, how are you?",
]

for q in questions:
    response = chain.invoke({"messages": [("user", q)]})
    print(f"\nQ: {q}")
    if response.tool_calls:
        for call in response.tool_calls:
            print(f"  -> wants to call: {call['name']}({call['args']})")
    else:
        print(f"  -> answers directly: {response.content}")