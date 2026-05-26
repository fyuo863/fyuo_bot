import os

from tools.base import GetWeatherTool, GetLocationTool, LetUserAnswer, ListFilesTool, ReadFileTool, DoCommand
from tools.agent_tool import AgentTool, GetModelList
from config import config

# 工作区：所有文件操作工具被限定在此目录内
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def main():
    # 1. 准备子工具
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

    # 2. 将 Agent 流程封装为工具
    agent_tool = AgentTool(
        sub_tools=sub_tools,
        model_label="fyuo-bot",
        max_depth=3,
    )
    agent_tool.workspace = WORKSPACE  # 设置工作区

    # 3. 手动调用触发 agent（显示在 execute 内部完成）
    agent_tool.execute(
        system="你是一个agent主管，可以调用工具获取信息，根据任务的复杂程度自行选择自己解决还是调用子agent解决，决定调用agent前要说明。",
        prompt="我这里今天天气怎么样？",
    )


if __name__ == "__main__":
    main()
