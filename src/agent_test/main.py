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

    # 3. 从命令行读入用户问题
    user_input = input("\n请输入你的问题: ").strip()
    if not user_input:
        print("问题不能为空")
        return

    # 4. 手动调用触发 agent（显示在 execute 内部完成）
    agent_tool.execute(
        system="你是一个agent主管，可以调用工具获取信息，根据任务的复杂程度自行选择自己解决还是调用子agent解决，决定调用agent前要说明。",
        prompt=user_input,
    )


if __name__ == "__main__":
    main()
