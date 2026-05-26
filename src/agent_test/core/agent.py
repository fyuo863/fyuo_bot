import json
from dataclasses import dataclass

from .chat import AgentChat, TextChunk, ReasoningChunk, ToolCall
from .usage import Usage
from tools.base import BaseTool


@dataclass
class ToolResult:
    name: str
    result: str


@dataclass
class UserInputRequired:
    question: str
    call_id: str


@dataclass
class AgentComplete:
    pass


class ReActAgent:

    def __init__(
        self,
        tools: list[BaseTool],
        system: str = "",
        model: str | None = None,
        workspace: str = "",
    ):
        self.tool_registry: dict[str, BaseTool] = {t.name: t for t in tools}
        for t in self.tool_registry.values():
            t.workspace = workspace
        self.system = system
        self.model = model
        self.messages: list[dict] = []

    def _tool_schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self.tool_registry.values()]

    def run(self, user_input: str, max_iterations: int = 20):
        self.messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user_input},
        ]
        return self._react_loop(max_iterations)

    def continue_conversation(self, user_input: str, max_iterations: int = 20):
        self.messages.append({"role": "user", "content": user_input})
        return self._react_loop(max_iterations)

    def _react_loop(self, max_iterations: int):
        for _ in range(max_iterations):
            stream_items: list = []
            for item in AgentChat.chat(
                messages=self.messages,
                tools=self._tool_schemas(),
                model=self.model,
            ):
                stream_items.append(item)
                yield item

            assistant_msg = Usage.assemble_assistant_message(stream_items)
            self.messages.append(assistant_msg)

            if "tool_calls" not in assistant_msg:
                yield AgentComplete()
                return

            for tc in assistant_msg["tool_calls"]:
                func_name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                tool = self.tool_registry.get(func_name)

                if tool is None:
                    result = f"错误：未找到工具 '{func_name}'"
                elif func_name == "let_user_answer":
                    yield UserInputRequired(
                        question=args.get("question", ""),
                        call_id=tc["id"],
                    )
                    return
                else:
                    result = tool.execute(**args)

                if func_name != "let_user_answer":
                    yield ToolResult(name=func_name, result=result)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": str(result),
                    })

        yield AgentComplete()

    def feed_user_response(self, user_response: str, call_id: str):
        self.messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": "用户已回复",
        })
        self.messages.append({
            "role": "user",
            "content": user_response,
        })

    def resume(self, max_iterations: int = 20):
        return self._react_loop(max_iterations)
