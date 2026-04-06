"""LangGraph workflow for a ReAct-style MCP tool-using agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from typing_extensions import Literal, NotRequired, TypedDict


REACT_SYSTEM_PROMPT = """You are a ReAct agent. Follow this loop strictly:
Thought -> Action (tool call) -> Observation -> Thought -> ...

Rules:
1. Use tools for factual information.
2. For multi-part questions, make multiple tool calls as needed.
3. Use tools for arithmetic when available.
4. Provide a Final Answer only after all required tool calls are done.
"""


class StepRecord(TypedDict):
	"""Single action-observation pair for the reasoning trace."""

	action: str
	action_input: dict[str, Any]
	observation: str


class PendingAction(TypedDict):
	"""Tool call chosen by the model and waiting for execution."""

	tool_name: str
	tool_args: dict[str, Any]
	tool_call_id: str


class ReActState(TypedDict):
	"""State schema required by the assignment and graph runtime."""

	input: str
	agent_scratchpad: str
	final_answer: str
	steps: list[StepRecord]
	messages: list[BaseMessage]
	is_final: bool
	pending_action: NotRequired[PendingAction | None]


def _append_scratchpad(existing: str, lines: Sequence[str]) -> str:
	clean_lines = [line for line in lines if line]
	if not clean_lines:
		return existing
	if existing.strip():
		return existing.strip() + "\n" + "\n".join(clean_lines)
	return "\n".join(clean_lines)


def create_mcp_client(
	project_root: Path,
	weather_url: str = "http://localhost:8000/mcp",
) -> MultiServerMCPClient:
	"""Create an MCP client using the provided assignment tool servers."""
	tools_dir = project_root / "Tools"
	return MultiServerMCPClient(
		{
			"math": {
				"command": sys.executable,
				"args": [str(tools_dir / "math_server.py")],
				"transport": "stdio",
			},
			"search": {
				"command": sys.executable,
				"args": [str(tools_dir / "search_server.py")],
				"transport": "stdio",
			},
			"weather": {
				"url": weather_url,
				"transport": "streamable_http",
			},
		}
	)


async def load_mcp_tools(
	mcp_client: MultiServerMCPClient,
	servers: Sequence[str],
) -> tuple[list[Any], dict[str, Any]]:
	"""Load tools from selected MCP servers and return both list and map."""
	tools: list[Any] = []
	for server in servers:
		server_tools = await mcp_client.get_tools(server_name=server)
		tools.extend(server_tools)
	return tools, {tool.name: tool for tool in tools}


def initialize_state(user_input: str) -> ReActState:
	"""Build initial graph state."""
	return {
		"input": user_input,
		"agent_scratchpad": "",
		"final_answer": "",
		"steps": [],
		"messages": [],
		"is_final": False,
		"pending_action": None,
	}


def create_react_graph(llm_with_tools: Any, tools_map: Mapping[str, Any], max_steps: int = 20):
	"""Create and compile a LangGraph ReAct workflow."""

	async def react_node(state: ReActState) -> ReActState:
		messages = list(state.get("messages", []))
		steps = list(state.get("steps", []))
		scratchpad = state.get("agent_scratchpad", "")

		if not messages:
			messages = [
				SystemMessage(content=REACT_SYSTEM_PROMPT),
				HumanMessage(content=state["input"]),
			]

		if len(steps) >= max_steps:
			capped = "Stopped because max reasoning steps were reached."
			scratchpad = _append_scratchpad(
				scratchpad,
				[
					"Thought: I should stop to avoid an infinite loop.",
					f"Final Answer: {capped}",
				],
			)
			return {
				**state,
				"messages": messages,
				"agent_scratchpad": scratchpad,
				"final_answer": capped,
				"is_final": True,
				"pending_action": None,
			}

		ai_message = await llm_with_tools.ainvoke(messages)
		messages.append(ai_message)

		tool_calls = getattr(ai_message, "tool_calls", None) or []
		if not tool_calls:
			final_answer = (ai_message.content or "").strip() or "No final answer generated."
			scratchpad = _append_scratchpad(
				scratchpad,
				[
					f"Thought: {(ai_message.content or '').strip() or 'I have enough information.'}",
					f"Final Answer: {final_answer}",
				],
			)
			return {
				**state,
				"messages": messages,
				"agent_scratchpad": scratchpad,
				"final_answer": final_answer,
				"is_final": True,
				"pending_action": None,
			}

		first_call = tool_calls[0]
		tool_name = first_call["name"]
		tool_args = first_call.get("args", {})
		tool_call_id = first_call.get("id", f"{tool_name}_call")

		scratchpad = _append_scratchpad(
			scratchpad,
			[
				f"Thought: {(ai_message.content or '').strip() or 'I need a tool to continue.'}",
				f"Action: {tool_name}({json.dumps(tool_args, ensure_ascii=True)})",
			],
		)

		return {
			**state,
			"messages": messages,
			"agent_scratchpad": scratchpad,
			"is_final": False,
			"final_answer": "",
			"pending_action": {
				"tool_name": tool_name,
				"tool_args": tool_args,
				"tool_call_id": tool_call_id,
			},
		}

	async def tool_node(state: ReActState) -> ReActState:
		pending = state.get("pending_action")
		if not pending:
			return {**state}

		tool_name = pending["tool_name"]
		tool_args = pending["tool_args"]
		tool_call_id = pending["tool_call_id"]

		if tool_name not in tools_map:
			observation = (
				f"Tool '{tool_name}' was requested but not found. "
				f"Available tools: {', '.join(sorted(tools_map.keys()))}"
			)
		else:
			try:
				observation = str(await tools_map[tool_name].ainvoke(tool_args))
			except Exception as exc:  # noqa: BLE001 - keep failure inside graph state
				observation = f"Tool execution error for '{tool_name}': {exc}"

		messages = list(state.get("messages", []))
		messages.append(ToolMessage(content=observation, tool_call_id=tool_call_id))

		steps = list(state.get("steps", []))
		steps.append(
			{
				"action": tool_name,
				"action_input": tool_args,
				"observation": observation,
			}
		)

		scratchpad = _append_scratchpad(
			state.get("agent_scratchpad", ""),
			[f"Observation: {observation}"],
		)

		return {
			**state,
			"messages": messages,
			"steps": steps,
			"agent_scratchpad": scratchpad,
			"pending_action": None,
		}

	def route_after_react(state: ReActState) -> Literal["tool", "end"]:
		return "end" if state.get("is_final", False) else "tool"

	builder = StateGraph(ReActState)
	builder.add_node("react_node", react_node)
	builder.add_node("tool_node", tool_node)

	builder.add_edge(START, "react_node")
	builder.add_conditional_edges(
		"react_node",
		route_after_react,
		{
			"tool": "tool_node",
			"end": END,
		},
	)
	builder.add_edge("tool_node", "react_node")

	return builder.compile()
