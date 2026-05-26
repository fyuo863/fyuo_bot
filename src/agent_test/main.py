import os
import uuid

from tools.base import GetWeatherTool, GetLocationTool, LetUserAnswer, ListFilesTool, ReadFileTool, DoCommand
from tools.agent_tool import AgentTool, GetModelList

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

SYSTEM_PROMPT = (
    "你是一个agent主管，根据任务的复杂程度自行选择自己解决还是调用子agent解决，"
    "可以调用工具获取信息，"
    "决定调用agent前要说明。"
)


def main():
    # 初始化记忆引擎（连接 Docker ChromaDB）

    sub_tools = [
        GetWeatherTool(),
        GetLocationTool(),
        LetUserAnswer(),
        ListFilesTool(),
        ReadFileTool(),
        DoCommand(),
        GetModelList(),
        AgentTool(),
    ]

    agent_tool = AgentTool(
        sub_tools=sub_tools,
        model_label="fyuo-bot",
        max_depth=3,
        # memory_engine=memory,
        session_id=str(uuid.uuid4())[:8],  # 每次启动一个唯一 session
    )
    agent_tool.workspace = WORKSPACE

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
