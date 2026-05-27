import os
import uuid

from tools.base import GetWeatherTool, GetLocationTool, LetUserAnswer, ListFilesTool, ReadFileTool, DoCommand, NewFileTool, WriteFileTool
from tools.agent_tool import AgentTool, GetModelList
from tools.memory_tools import ReplaceMemoryTool, GetHistoryTool
from memory import MemoryManager, HistoryManager

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ---- 加载持久记忆 ----
memory_manager = MemoryManager(WORKSPACE)
memory_snapshot = memory_manager.get_snapshot()

# ---- 初始化历史记录 (SQLite) ----
history_manager = HistoryManager(WORKSPACE)

SYSTEM_PROMPT = (
    "你是一个agent主管，根据任务的复杂程度自行选择自己解决还是调用子agent解决，"
    "可以调用工具获取信息，"
    "决定调用agent前要说明。\n"
    "\n"
    "你有两套存储系统，职责严格区分：\n"
    "  - replace_memory（MEMORY.md / USER.md）：只存用户相关的持久信息。\n"
    "    MEMORY 只存用户操作习惯（如\"喜欢先计划再动手\"、\"回复用纯文本\"）；\n"
    "    USER 只存用户个人信息（如饮食偏好、使用系统、职业等）。\n"
    "    严禁存入：任务执行记录、代码细节、文件路径、项目事实。\n"
    "  - get_history（SQLite）：对话历史，每轮自动浓缩存入，按话题可搜索。\n"
    "记录高信息密度的精简短句。当添加记忆超出容量时，先查看现有内容，"
    "合并相似条目、精简废话后再重试。\n"
    "\n"
    "【多轮自评机制】每次回答后系统会自动要求你自我评价。"
    "请在自评时审视：回答是否完整？是否有事实错误？是否可以进一步优化？"
    "如果发现问题，主动调用工具修正；确认无误后回复'确认完成'以结束任务。\n"
    "\n"
    "=== 以下是从上次会话继承的持久记忆（本次会话固定不变） ===\n"
    + (memory_snapshot if memory_snapshot else "暂无历史记忆。")
)


def main():
    # 创建记忆/历史工具并注入管理器
    replace_memory_tool = ReplaceMemoryTool()
    replace_memory_tool.memory_manager = memory_manager
    get_history_tool = GetHistoryTool()
    get_history_tool.history_manager = history_manager

    sub_tools = [
        GetWeatherTool(),
        GetLocationTool(),
        LetUserAnswer(),
        NewFileTool(),
        WriteFileTool(),
        ListFilesTool(),
        ReadFileTool(),
        DoCommand(),
        GetModelList(),
        replace_memory_tool,
        get_history_tool,
        AgentTool(),
    ]

    agent_tool = AgentTool(
        sub_tools=sub_tools,
        model_label="fyuo-bot",
        max_depth=3,
        auto_reflect=True,
        max_reflections=3,
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

        result = agent_tool.chat(system=SYSTEM_PROMPT, prompt=user_input, model="deepseek-v4-flash")
        if result:
            history_manager.save_turn(agent_tool.session_id, user_input, result)


if __name__ == "__main__":
    main()
