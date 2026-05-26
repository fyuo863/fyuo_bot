import os
import uuid

from tools.base import GetWeatherTool, GetLocationTool, LetUserAnswer, ListFilesTool, ReadFileTool, DoCommand
from tools.agent_tool import AgentTool, GetModelList
from tools.memory_tools import AddMemoryTool, RemoveMemoryTool, GetMemoryTool
from memory import MemoryManager

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ---- 加载持久记忆 ----
memory_manager = MemoryManager(WORKSPACE)
memory_snapshot = memory_manager.get_snapshot()

SYSTEM_PROMPT = (
    "你是一个agent主管，根据任务的复杂程度自行选择自己解决还是调用子agent解决，"
    "可以调用工具获取信息，"
    "决定调用agent前要说明。\n"
    "\n"
    "你有持久记忆能力，可以使用 add_memory / remove_memory / get_memory 工具管理记忆。"
    "记录高信息密度的精简短句。当添加记忆超出容量时，先查看现有内容，"
    "合并相似条目、精简废话后再重试。\n"
    "\n"
    "=== 以下是从上次会话继承的持久记忆（本次会话固定不变） ===\n"
    + (memory_snapshot if memory_snapshot else "暂无历史记忆。")
)


def main():
    # 创建记忆工具并注入 MemoryManager
    add_memory_tool = AddMemoryTool()
    add_memory_tool.memory_manager = memory_manager
    remove_memory_tool = RemoveMemoryTool()
    remove_memory_tool.memory_manager = memory_manager
    get_memory_tool = GetMemoryTool()
    get_memory_tool.memory_manager = memory_manager

    sub_tools = [
        GetWeatherTool(),
        GetLocationTool(),
        LetUserAnswer(),
        ListFilesTool(),
        ReadFileTool(),
        DoCommand(),
        GetModelList(),
        add_memory_tool,
        remove_memory_tool,
        get_memory_tool,
        AgentTool(),
    ]

    agent_tool = AgentTool(
        sub_tools=sub_tools,
        model_label="fyuo-bot",
        max_depth=3,
        session_id=str(uuid.uuid4())[:8],
    )
    agent_tool.workspace = WORKSPACE

    print(f"已加载持久记忆 ({len(memory_snapshot)} 字符)")

    while True:
        user_input = input("\nUser: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("已退出")
            break

        agent_tool.chat(system=SYSTEM_PROMPT, prompt=user_input)


if __name__ == "__main__":
    main()
