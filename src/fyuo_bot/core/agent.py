import json
from dataclasses import dataclass

from .chat import AgentChat, TextChunk, ReasoningChunk, ToolCall, count_tokens
from .usage import Usage
from tools.base import BaseTool

DIM = "\033[2m"
RESET = "\033[0m"


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
    final_text: str = ""


class ReActAgent:

    COMPRESSION_THRESHOLD = 1_000_000   # 超过 100 万 token 触发压缩
    KEEP_RECENT_MESSAGES = 6            # 压缩时保留最近 N 条消息

    def __init__(
        self,
        tools: list[BaseTool],
        system: str = "",
        model: str | None = None,
        workspace: str = "",
        auto_reflect: bool = False,
        max_reflections: int = 3,
    ):
        self.tool_registry: dict[str, BaseTool] = {t.name: t for t in tools}
        for t in self.tool_registry.values():
            t.workspace = workspace
        self.system = system
        self.model = model
        self.messages: list[dict] = []
        self.auto_reflect = auto_reflect
        self.max_reflections = max_reflections
        self._compression_count = 0

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

    SELF_REFLECT_PROMPT = (
        "【自动自评】请审视你刚才的回答是否完整、准确。"
        "如果发现错误或遗漏，请调用工具修正或补充；"
        "如果确认回答已完美完成，请直接回复'确认完成'。"
    )

    DONE_MARKERS = ("确认完成", "无需改进", "任务完成", "已完成")

    def _react_loop(self, max_iterations: int):
        reflections = 0
        pending_answer = ""

        for i in range(max_iterations):
            token_n = count_tokens(self.messages)

            # 上下文过长时触发压缩
            if token_n > self.COMPRESSION_THRESHOLD:
                before = token_n
                self._compress_context()
                token_n = count_tokens(self.messages)
                print(f"\n{DIM}[上下文压缩] {before} → {token_n} tokens{RESET}")

            print(f"\n{DIM}[轮次 {i+1}] 上下文: {token_n} tokens{RESET}")

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
                content = assistant_msg.get("content", "")

                if self.auto_reflect and reflections < self.max_reflections:
                    if self._is_done_marker(content):
                        final = pending_answer or content
                        yield AgentComplete(final_text=final)
                        return

                    reflections += 1
                    pending_answer = content
                    self.messages.append({
                        "role": "user",
                        "content": self.SELF_REFLECT_PROMPT,
                    })
                    continue

                final = pending_answer or content
                yield AgentComplete(final_text=final)
                return

            # 自评后 agent 选择调用工具改进 → 清除暂存答案
            pending_answer = ""

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

    def _compress_context(self):
        """压缩消息历史：保留 system + 最近消息，中间部分用 LLM 浓缩摘要替代。"""
        system_msg = self.messages[0] if self.messages and self.messages[0]["role"] == "system" else None
        start = 1 if system_msg else 0
        keep = self.KEEP_RECENT_MESSAGES

        if len(self.messages) - start <= keep:
            return  # 消息不够多，无需压缩

        recent = self.messages[-keep:]
        old = self.messages[start:-keep]

        summary = AgentChat.summarize_context(old, model=self.model)
        if not summary:
            return  # 压缩失败，不修改消息列表

        self._compression_count += 1
        new_messages = []
        if system_msg:
            new_messages.append(system_msg)
        new_messages.append({
            "role": "user",
            "content": (
                f"[上下文摘要 #{self._compression_count}]\n"
                f"以下是之前对话的关键要点摘要，请据此继续工作：\n\n{summary}"
            ),
        })
        new_messages.extend(recent)
        self.messages = new_messages

    def _is_done_marker(self, text: str) -> bool:
        """检测文本是否仅为完成确认信号（不含实质内容）。"""
        stripped = text.strip()
        if len(stripped) > 30:
            return False
        return any(marker in stripped for marker in self.DONE_MARKERS)

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
