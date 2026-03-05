from __future__ import annotations

import argparse
from typing import Any

from eidetic.integrations.langchain import EideticLangChainMemory

try:
    from langchain.agents import create_agent
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "LangChain dependencies are missing. "
        'Run: uv run --extra langchain python examples/langchain_agent_chat.py'
    ) from exc


class MemoryAwareDemoModel(BaseChatModel):
    """A local deterministic model for LangChain agent demo without API keys."""

    @property
    def _llm_type(self) -> str:
        return "memory-aware-demo"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = str(messages[-1].content) if messages else ""
        user_input, memory_context = self._parse_payload(content)
        reply = self._compose_reply(user_input=user_input, memory_context=memory_context)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

    @staticmethod
    def _parse_payload(content: str) -> tuple[str, str]:
        user_input = ""
        memory_context = ""

        if "[USER_INPUT]" in content and "[MEMORY_CONTEXT]" in content:
            user_part = content.split("[USER_INPUT]", 1)[1]
            parts = user_part.split("[MEMORY_CONTEXT]", 1)
            user_input = parts[0].strip()
            memory_context = parts[1].strip() if len(parts) > 1 else ""
            return user_input, memory_context

        return content.strip(), ""

    @staticmethod
    def _compose_reply(user_input: str, memory_context: str) -> str:
        if not user_input:
            if memory_context and memory_context != "(empty)":
                top = memory_context.splitlines()[0]
                return f"你没有输入新消息。我当前记得的关键内容是：{top}"
            return "你没有输入新消息，目前还没有可用记忆。"

        if memory_context and memory_context != "(empty)":
            first = memory_context.splitlines()[0]
            return (
                f"收到：{user_input}\n"
                f"结合历史记忆，我还记得：{first}\n"
                "如果你愿意，我可以继续基于这段记忆追问或总结。"
            )
        return f"收到：{user_input}\n这是我们当前会话中的第一批有效记忆，我会记住它。"


def build_agent():
    model = MemoryAwareDemoModel()
    return create_agent(
        model=model,
        tools=[],
        system_prompt=(
            "You are a concise assistant in a memory demo. "
            "Use the provided memory context when available."
        ),
    )


def run_turn(agent, memory: EideticLangChainMemory, user_input: str) -> str:
    memory_vars = memory.load_memory_variables({memory.input_key: user_input})
    context = memory_vars.get(memory.memory_key, "")
    payload = (
        f"[USER_INPUT]\n{user_input}\n"
        f"[MEMORY_CONTEXT]\n{context if context else '(empty)'}"
    )
    output = agent.invoke({"messages": [{"role": "user", "content": payload}]})
    reply = str(output["messages"][-1].content)

    if user_input.strip():
        memory.save_context({memory.input_key: user_input}, {"output": reply})
    return reply


def interactive_chat(system: str, mode: str) -> None:
    memory = EideticLangChainMemory(
        system=system,
        config={"plugin_config": {"mode": mode}},
        input_key="input",
        memory_key="history",
        top_k=5,
        session_tag="demo-chat",
    )
    agent = build_agent()

    print("LangChain + Eidetic chat demo")
    print("commands: /exit | /memory | /clear")
    print("tip: press Enter on empty line to test no-message recall")
    print("-" * 64)

    while True:
        user_input = input("you> ")
        cmd = user_input.strip().lower()
        if cmd == "/exit":
            print("bye.")
            break
        if cmd == "/clear":
            memory.clear()
            print("agent> memory cleared")
            continue
        if cmd == "/memory":
            mem = memory.load_memory_variables({memory.input_key: ""})
            history = mem.get(memory.memory_key, "")
            print("agent-memory>")
            print(history if history else "(empty)")
            continue

        reply = run_turn(agent, memory, user_input)
        print(f"agent> {reply}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LangChain agent chat demo with Eidetic external memory."
    )
    parser.add_argument("--system", default="letta", choices=["letta", "graphrag", "lightrag", "hipporag"])
    parser.add_argument("--mode", default="mock", choices=["mock", "native", "auto"])
    args = parser.parse_args()
    interactive_chat(system=args.system, mode=args.mode)


if __name__ == "__main__":
    main()
