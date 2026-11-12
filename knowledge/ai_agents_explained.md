# AI Agents Explained

## What Is an AI Agent?
An AI agent is a system that uses a large language model (LLM) as its "brain" to decide what to do next in order to reach a goal. Unlike a basic chatbot that only replies from what it already knows, an agent can take actions: it can call tools, look up information, run calculations, and then use the results to decide its next step. This loop of thinking, acting and observing continues until the agent decides the task is complete.

## The Core Components
Most AI agents are built from four components:
1. Perception: receiving input, such as a user question or data from the environment.
2. Planning and reasoning: the LLM breaks the goal into steps and chooses which action to take.
3. Tools: functions the agent can call, such as a search engine, a database query, a calculator, or an API.
4. Memory: short-term memory holds the current conversation; long-term memory stores information across sessions so the agent can remember past interactions.

## The ReAct Pattern
ReAct stands for "Reason + Act". In each cycle the agent reasons about the problem, chooses an action (a tool call), observes the result, and then reasons again with the new information. This pattern reduces hallucination because the agent grounds its answers in real data returned by tools rather than guessing.

## Retrieval-Augmented Generation (RAG)
RAG is a technique where relevant documents are retrieved from a knowledge base and added to the LLM's prompt before it answers. Documents are split into chunks, each chunk is converted into a vector embedding, and the embeddings are stored in a vector database. When a question arrives, it is embedded too, and a vector search finds the chunks with the most similar meaning. In an agentic system, RAG is usually just one of several tools the agent can choose to use.

## Tool Calling
Modern LLMs support tool calling (also called function calling). The developer describes each tool with a name, a description and its parameters. The LLM does not run the tool itself; instead it outputs a structured request saying which tool to call and with which arguments. The application runs the tool and sends the result back to the LLM. Clear tool descriptions are critical, because the LLM uses them to decide which tool fits the user's request.

## Common Challenges
Agents can get stuck in loops, call the wrong tool, or run up high costs from many LLM calls. Good practices include limiting the number of steps, writing precise tool descriptions, logging every step for debugging, and keeping a human in the loop for risky actions.
