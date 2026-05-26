import os

from tools.base import GetWeatherTool, GetLocationTool, LetUserAnswer, ListFilesTool, ReadFileTool, DoCommand
from tools.agent_tool import AgentTool, GetModelList

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

SYSTEM_PROMPT = (
    "你是一个agent主管，你的职责是根据用户提问制定执行计划，并按照计划内容的复杂程度调用子agent执行。"
    "子agent可以调用工具获取信息，"
    "决定调用agent前要说明。"
)


def main():
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
    )
    agent_tool.workspace = WORKSPACE

    while True:
        user_input = input("\nUser: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("已退出")
            break

        # chat() 首次自动创建 agent，后续复用历史 → 多轮记忆
        agent_tool.chat(system=SYSTEM_PROMPT, prompt=user_input)


if __name__ == "__main__":
    main()
