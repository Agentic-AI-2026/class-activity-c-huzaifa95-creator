[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/LWzSlHLS)
# Quiz: Convert ReAct Agent to LangGraph 🦜🕸️

## Objective
Convert a standard working **ReAct agent** (implemented in LangChain) into a **LangGraph workflow**. Your implementation must preserve the iterative reasoning and tool-usage behavior inherent to the ReAct framework.

---

## 🛠 Provided Resources
* **Existing ReAct agent code** (LangChain-based).
* **Tool implementations** (functional and ready for use).

---

## 📋 Requirements

### 1. Define State
Create a state structure (TypedDict or Pydantic) to represent the workflow. Your state must include:
* `input`: The original user query.
* `agent_scratchpad`: Stores intermediate reasoning (Thoughts, Actions, Observations).
* `final_answer`: The final response delivered to the user.
* `steps`: (Optional) A list to track history of actions and observations.

### 2. ReAct Node (Reasoning + Action)
Implement a node that:
1.  Takes the current state.
2.  Calls the LLM using **ReAct-style prompting**.
3.  Produces either an **Action** (tool name + arguments) or a **Final Answer**.
4.  Updates the state accordingly.

### 3. Tool Execution Node
Implement a node that:
1.  Executes the tool selected by the ReAct node.
2.  Passes the correct arguments to the tool.
3.  Stores the **Observation** (result) back in the state.
4.  Updates the scratchpad to prepare for the next reasoning step.

### 4. Graph Flow
Construct a LangGraph workflow that follows this logic:

> **START** $\rightarrow$ `react_node` $\rightarrow$ **Conditional Edge**
> * If **Action** $\rightarrow$ `tool_node` $\rightarrow$ `react_node`
> * If **Final Answer** $\rightarrow$ **END**

**The graph must:**
* Support iterative reasoning loops.
* Continue execution until a terminal state (Final Answer) is reached.

### 5. Conditional Routing
Implement the router logic to determine the next step based on the model's output:
- `is_action` $\rightarrow$ Route to `tool_node`.
- `is_final` $\rightarrow$ Route to `END`.

---

## 🧪 Test Case
Your implementation should successfully process complex, multi-step queries such as:

> *"What is the weather in Lahore and who is the current Prime Minister of Pakistan? Now get the age of PM and tell us will this weather suits PM health."*

---

## ⚠️ Constraints
* **No Hardcoding:** Do not hardcode outputs; the logic must be dynamic.
* **Reasoning Integrity:** Maintain the "Thought $\rightarrow$ Action $\rightarrow$ Observation" flow.
* **Scalability:** The agent must be capable of calling tools multiple times in a single run.
* **State Management:** Ensure proper state updates to prevent infinite loops or data loss between iterations.

---

## 🚀 Submission
Push your solution to this repository using the following structure:
```text
.
├── main.py          # Entry point for execution
├── graph.py         # LangGraph definition (optional)
└── README.md        # Project documentation
