"""Entry point for the ReAct-to-LangGraph assignment solution."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Sequence

from langchain_groq import ChatGroq

from graph import create_mcp_client, create_react_graph, initialize_state, load_mcp_tools


DEFAULT_QUERY = (
	"What is the weather in Lahore and who is the current Prime Minister of Pakistan? "
	"Now get the age of PM and tell us will this weather suits PM health."
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run the LangGraph ReAct workflow.")
	parser.add_argument(
		"query",
		nargs="?",
		default=DEFAULT_QUERY,
		help="User query to process",
	)
	parser.add_argument(
		"--servers",
		default="math,search,weather",
		help="Comma-separated MCP servers to enable (example: math,search,weather)",
	)
	parser.add_argument(
		"--max-steps",
		type=int,
		default=20,
		help="Max graph loops before forcing stop",
	)
	parser.add_argument(
		"--model",
		default="llama-3.3-70b-versatile",
		help="Groq chat model name",
	)
	return parser.parse_args()


def parse_servers(raw: str) -> list[str]:
	return [item.strip() for item in raw.split(",") if item.strip()]


async def run_workflow(
	query: str,
	servers: Sequence[str],
	max_steps: int,
	model_name: str,
) -> dict:
	api_key = os.getenv("GROQ_API_KEY", "").strip()
	if not api_key:
		raise RuntimeError(
			"GROQ_API_KEY is missing. Set it in your environment before running main.py."
		)

	project_root = Path(__file__).resolve().parent

	mcp_client = create_mcp_client(project_root)
	tools, tools_map = await load_mcp_tools(mcp_client, servers)

	if not tools:
		raise RuntimeError(
			"No tools were loaded from MCP servers. Check --servers and tool server setup."
		)

	llm = ChatGroq(model=model_name, temperature=0, api_key=api_key)
	llm_with_tools = llm.bind_tools(tools)

	app = create_react_graph(llm_with_tools=llm_with_tools, tools_map=tools_map, max_steps=max_steps)
	initial_state = initialize_state(query)
	return await app.ainvoke(initial_state)


def print_result(result: dict) -> None:
	final_answer = result.get("final_answer", "")
	steps = result.get("steps", [])

	print("\n=== FINAL ANSWER ===")
	print(final_answer or "No final answer produced.")

	print("\n=== STEPS ===")
	if not steps:
		print("No tool steps recorded.")
		return

	for index, step in enumerate(steps, start=1):
		print(f"{index}. Action: {step.get('action')}")
		print(f"   Args: {step.get('action_input')}")
		print(f"   Observation: {step.get('observation')}")


async def _main_async() -> None:
	args = parse_args()
	servers = parse_servers(args.servers)
	result = await run_workflow(
		query=args.query,
		servers=servers,
		max_steps=args.max_steps,
		model_name=args.model,
	)
	print_result(result)


if __name__ == "__main__":
	asyncio.run(_main_async())
