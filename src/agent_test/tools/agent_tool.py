"""
AgentTool —— 将整个 ReAct Agent 封装为一个 BaseTool，支持嵌套。

深度控制:
  - max_depth 在构造时设定，不暴露给模型（模型无法篡改）
  - 每嵌套一层，子 AgentTool 的 max_depth 自动减 1
  - 深度归零时，最内层 agent 被强制以最简单方式完成任务
"""

from dataclasses import dataclass, field

from .base import BaseTool
from core.agent import (
    ReActAgent,
    TextChunk,
    ReasoningChunk,
    ToolCall,
    ToolResult,
    UserInputRequired,
    AgentComplete,
)

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


@dataclass
class AgentTool(BaseTool):
    """将 ReAct Agent 封装为工具，支持嵌套调用。

    参数:
        sub_tools:    子 agent 可用的工具列表
        model_label:  显示用的模型名称
        max_depth:    内部深度控制，模型不可见 (默认 3)
    """

    sub_tools: list[BaseTool] = field(default_factory=list)
    model_label: str = "AI"
    max_depth: int = 3

    name: str = "run_agent"
    description: str = (
        "启动一个子 Agent 处理复杂任务。当你需要多步骤推理、"
        "调用多个工具协作完成一个目标时使用。"
    )
    parameters: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "system": {
                "type": "string",
                "description": "子 Agent 的系统提示词",
            },
            "prompt": {
                "type": "string",
                "description": "需要子 Agent 处理的任务描述",
            },
        },
        "required": ["system", "prompt"],
    })

    def execute(self, system: str = "", prompt: str = "", **kwargs) -> str:
        """运行完整 ReAct 循环。深度由 self.max_depth 内部控制，模型无法传入。"""
        print(f"\n{YELLOW}[嵌套深度] 当前层剩余 {self.max_depth} 层{RESET}")

        if self.max_depth <= 0:
            return "已达到最大嵌套深度，无法继续调用子 Agent。"

        child_depth = self.max_depth - 1

        # 为子 agent 准备工具：如果子工具中有 AgentTool，克隆并降低深度
        child_tools: list[BaseTool] = []
        for t in self.sub_tools:
            if isinstance(t, AgentTool):
                child_tools.append(AgentTool(
                    sub_tools=t.sub_tools,
                    model_label=t.model_label,
                    max_depth=child_depth,
                ))
            else:
                child_tools.append(t)

        # 最内层：注入简化指令，强制直接完成
        if child_depth <= 0:
            system = (
                "【深度警告】你已到达最大嵌套深度，必须直接完成任务，"
                "禁止调用 run_agent 或其他子 Agent 工具。用最简单直接的方式回复用户。"
                + system
            )

        print()
        print("=" * 50)
        print(f"User:  {prompt}")
        print("=" * 50)

        agent = ReActAgent(tools=child_tools, system=system)
        return self._run_loop(agent, prompt, is_resume=False)

    def _run_loop(self, agent: ReActAgent, user_input: str, is_resume: bool) -> str:
        """执行一次 ReAct 循环（含中断恢复），返回最终文本。"""
        gen = agent.run(user_input) if not is_resume else agent.resume()

        reasoning_printed = False
        final_text = ""

        for event in gen:
            if isinstance(event, ReasoningChunk):
                if not reasoning_printed:
                    print(f"\n{DIM}[思考过程]", end="", flush=True)
                    reasoning_printed = True
                print(event.content, end="", flush=True)

            elif isinstance(event, TextChunk):
                if reasoning_printed:
                    print(f"{RESET}\n{self.model_label}:    {BOLD}", end="", flush=True)
                    reasoning_printed = False
                print(event.content, end="", flush=True)
                final_text += event.content

            elif isinstance(event, ToolCall):
                print(f"\n{GREEN}[调用工具] {event.name}{RESET}")

            elif isinstance(event, ToolResult):
                print(f"{GREEN}[工具返回] {event.name}: {event.result}{RESET}")

            elif isinstance(event, UserInputRequired):
                question = event.question
                print(f"\n{GREEN}[Agent 提问] {question}{RESET}")
                user_response = input(f"{BOLD}你的回答: {RESET}")

                agent.feed_user_response(user_response, event.call_id)
                print("-" * 50)
                return self._run_loop(agent, user_response, is_resume=True)

            elif isinstance(event, AgentComplete):
                print(RESET)
                print("-" * 50)
                print("任务圆满完成！")

        return final_text
