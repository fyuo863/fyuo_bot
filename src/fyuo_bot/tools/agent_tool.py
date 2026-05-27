"""
AgentTool —— 将整个 ReAct Agent 封装为一个 BaseTool，支持嵌套、模型选择、记忆持久化。
"""

from dataclasses import dataclass, field

from config import config
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
CYAN = "\033[36m"
RESET = "\033[0m"


@dataclass
class AgentTool(BaseTool):

    sub_tools: list[BaseTool] = field(default_factory=list)
    model_label: str = "AI"
    max_depth: int = 3
    auto_reflect: bool = False
    max_reflections: int = 3
    memory_engine: object = None   # MemoryEngine 实例，可选
    session_id: str = ""   # 记忆的 session 标识

    name: str = "run_agent"
    description: str = (
        "启动一个子 Agent 处理复杂任务。当你需要多步骤推理、"
        "调用多个工具协作完成一个目标时使用。调用前先调 get_model_list 了解可用模型，"
        "根据任务复杂度选择 model"
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
            "model": {
                "type": "string",
                "description": "指定子 Agent 使用的模型名称",
            },
        },
        "required": ["system", "prompt"],
    })

    def __post_init__(self):
        self._agent: ReActAgent | None = None

    def execute(self, system: str = "", prompt: str = "", model: str = "", **kwargs) -> str:
        """单次调用（工具模式）。"""
        effective_model = model if model in config.MODEL_REGISTRY else None
        model_info = config.MODEL_REGISTRY.get(model, config.MODEL_REGISTRY["deepseek-v4-pro"])
        self.model_label = model_info["name"]
        print(f"\n{YELLOW}[嵌套深度] 当前层剩余 {self.max_depth} 层 | 模型: {self.model_label}{RESET}")
        if self.max_depth <= 0:
            return "已达到最大嵌套深度，无法继续调用子 Agent。"
        child_depth = self.max_depth - 1
        child_tools: list[BaseTool] = []
        for t in self.sub_tools:
            if isinstance(t, AgentTool):
                child_tools.append(AgentTool(sub_tools=t.sub_tools, model_label=t.model_label, max_depth=child_depth, auto_reflect=self.auto_reflect, max_reflections=self.max_reflections))
            else:
                child_tools.append(t)
        if child_depth <= 0:
            system = ("【深度警告】你已到达最大嵌套深度，必须直接完成任务。" + system)
        print()
        print("=" * 50)
        print(f"User:  {prompt}")
        print("=" * 50)
        agent = ReActAgent(tools=child_tools, system=system, model=effective_model, workspace=self.workspace, auto_reflect=self.auto_reflect, max_reflections=self.max_reflections)
        result = self._run_loop(agent, prompt, is_resume=False)
        self._save_task_memory(prompt, result)
        return result

    def chat(self, system: str, prompt: str, model: str = "") -> str:
        """多轮对话模式：带记忆检索和持久化。"""
        # ---- 记忆检索 ----
        memory_context = ""
        if self.memory_engine:
            memory_context = self.memory_engine.retrieve_memory(prompt)
            if memory_context:
                print(f"{CYAN}[记忆] 已加载相关历史记忆{RESET}")

        enriched_system = system + memory_context

        if self._agent is None:
            effective_model = model if model in config.MODEL_REGISTRY else None
            model_info = config.MODEL_REGISTRY.get(model, config.MODEL_REGISTRY["deepseek-v4-pro"])
            self.model_label = model_info["name"]
            child_depth = self.max_depth - 1
            child_tools = self._build_child_tools(child_depth)
            active_system = enriched_system
            if child_depth <= 0:
                active_system = ("【深度警告】你已到达最大嵌套深度，必须直接完成任务。" + enriched_system)
            self._agent = ReActAgent(tools=child_tools, system=active_system, model=effective_model, workspace=self.workspace, auto_reflect=self.auto_reflect, max_reflections=self.max_reflections)
            gen = self._agent.run(prompt)
        else:
            gen = self._agent.continue_conversation(prompt)

        final_text = self._run_loop_direct(gen)

        # ---- 记忆持久化（自动总结后存入） ----
        if final_text:
            self._save_task_memory(prompt, final_text)

        return final_text

    def _summarize_task(self, user_input: str, agent_response: str) -> str:
        """调用 LLM 将本次任务总结为一句话。"""
        from openai import OpenAI
        from config.config import api_key, base_url, model_name
        client = OpenAI(api_key=api_key, base_url=base_url)
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一个任务总结器。将用户的任务和agent的完成情况用一句简洁的话总结"
                            "（不超过50字），只返回总结本身，不要加任何前缀或引号。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"用户任务：{user_input}\nAgent完成情况：{agent_response[:2000]}",
                    },
                ],
                temperature=0.3,
                stream=False,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[记忆] 总结任务失败: {e}")
            return agent_response[:200]

    def _save_task_memory(self, user_input: str, agent_response: str):
        """总结任务并存入记忆。"""
        if not self.memory_engine:
            return
        import time
        summary = self._summarize_task(user_input, agent_response)
        if summary:
            sid = self.session_id or str(int(time.time()))
            self.memory_engine.save_memory(sid, user_input, summary)

    def _build_child_tools(self, child_depth: int) -> list[BaseTool]:
        child_tools: list[BaseTool] = []
        for t in self.sub_tools:
            if isinstance(t, AgentTool):
                child_tools.append(AgentTool(sub_tools=t.sub_tools, model_label=t.model_label, max_depth=child_depth, auto_reflect=self.auto_reflect, max_reflections=self.max_reflections))
            else:
                child_tools.append(t)
        return child_tools

    def _run_loop(self, agent: ReActAgent, user_input: str, is_resume: bool) -> str:
        gen = agent.run(user_input) if not is_resume else agent.resume()
        return self._run_loop_direct(gen)

    def _run_loop_direct(self, gen) -> str:
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
                self._agent.feed_user_response(user_response, event.call_id)
                print("-" * 50)
                return self._run_loop_direct(self._agent.resume())
            elif isinstance(event, AgentComplete):
                if event.final_text:
                    final_text = event.final_text
                print(RESET)
                print("-" * 50)
                print("任务圆满完成！")

        return final_text


@dataclass
class GetModelList(BaseTool):

    name: str = "get_model_list"
    description: str = "获取可用的模型列表，了解各模型的能力（简单/强力），在调用 run_agent 前使用"
    parameters: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {},
        "required": [],
    })

    def execute(self, **kwargs) -> str:
        lines = ["可用模型列表："]
        for info in config.MODEL_REGISTRY.values():
            lines.append(f"  - {info['name']} ({info['tier']}): {info['description']}")
        lines.append("建议：简单/快速任务选简易模型，复杂推理/代码/数学选复杂模型。")
        return "\n".join(lines)
