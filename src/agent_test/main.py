from core.chat import AgentChat, TextChunk, ToolCall
from tools.base import GetWeatherTool, GetLocationTool


def main():
    # 准备工具
    weather_tool = GetWeatherTool()
    location_tool = GetLocationTool()
    tools = [weather_tool.to_openai_schema(), location_tool.to_openai_schema()]

    input = "我这里今天天气怎么样？"

    print("=" * 50)
    print(f"User:  {input}")
    print("=" * 50)

    # 第一轮：发送用户问题 + 工具定义，模型可能返回 tool call 意图
    for item in AgentChat.chat(
        input,
        system="你是一个生活助手，可以调用工具获取信息。",
        tools=tools,
    ):
        if isinstance(item, TextChunk):
            print(item.content, end="", flush=True)

        elif isinstance(item, ToolCall):
            print(f"\n[调用工具] {item.name}({item.arguments})")

            # 执行工具
            import json
            args = json.loads(item.arguments)
            if item.name == weather_tool.name:
                result = weather_tool.execute(**args)
            elif item.name == location_tool.name:
                result = location_tool.execute(**args)
            print(f"[工具结果] {result}")

            # 第二轮：把工具结果喂回模型，获取最终回复
            print("=" * 50)
            print("AI:    ", end="", flush=True)
            for chunk in AgentChat.chat(
                f"工具 {item.name} 返回了：{result}，请用中文把结果告诉用户",
                system="你是一个生活助手，根据工具结果简洁回复用户。",
            ):
                if isinstance(chunk, TextChunk):
                    print(chunk.content, end="", flush=True)
            print()

    print()


if __name__ == "__main__":
    main()
