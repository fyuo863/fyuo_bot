from tools.base import GetWeatherTool, GetLocationTool, LetUserAnswer
from tools.agent_tool import AgentTool


def main():
    # 1. 准备子工具
    sub_tools = [
        GetWeatherTool(),
        GetLocationTool(),
        LetUserAnswer(),
    ]

    # 2. 将 Agent 流程封装为工具（max_depth 内部控制，模型不可见）
    agent_tool = AgentTool(
        sub_tools=sub_tools,
        model_label="DeepSeek",
        max_depth=3,
    )

    # 3. 手动调用触发 agent
    result = agent_tool.execute(
        system="你是一个生活助手，可以调用工具获取信息。",
        prompt="我这里今天天气怎么样？",
    )

    if result:
        print(f"\n最终结果: {result}")


if __name__ == "__main__":
    main()
