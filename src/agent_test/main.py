from tools.base import GetWeatherTool, GetLocationTool, LetUserAnswer
from tools.agent_tool import AgentTool, GetModelList
from config import config


def main():
    # 1. 准备子工具
    sub_tools = [
        GetWeatherTool(),
        GetLocationTool(),
        LetUserAnswer(),
        GetModelList(),
        AgentTool(),
    ]

    # 2. 将 Agent 流程封装为工具（max_depth 内部控制，模型不可见）
    agent_tool = AgentTool(
        sub_tools=sub_tools,
        model_label="fyuo-bot",  # 显示用标签，实际模型由 execute 时参数控制
        max_depth=3,
    )

    # 3. 手动调用触发 agent（显示在 execute 内部完成，不需要再打印返回值）
    agent_tool.execute(
        # system="你是一个agent主管，可以调用工具获取信息，任何任务必须使用子agent执行，决定调用agent前要说明。",
        system="你是一个agent主管，可以调用工具获取信息，根据任务的复杂程度自行选择自己解决还是调用子agent解决，决定调用agent前要说明。",
        prompt="我这里今天天气怎么样？",
    )


if __name__ == "__main__":
    main()
