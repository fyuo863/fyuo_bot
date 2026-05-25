from core.chat import TextChunk, ReasoningChunk, ToolCall
from core.agent import ReActAgent, ToolResult, AgentComplete
from tools.base import GetWeatherTool, GetLocationTool
from config.config import api_key, base_url, model_name

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RESET = "\033[0m"


def main():
    agent = ReActAgent(
        tools=[GetWeatherTool(), GetLocationTool()],
        system="你是一个生活助手，可以调用工具获取信息。",
    )

    input_text = "我这里今天天气怎么样？"
    print("=" * 50)
    print(f"User:  {input_text}")
    print("=" * 50)

    reasoning_printed = False
    for event in agent.run(input_text):
        if isinstance(event, ReasoningChunk):
            if not reasoning_printed:
                print(f"\n{DIM}[思考过程]", end="", flush=True)
                reasoning_printed = True
            print(event.content, end="", flush=True)

        elif isinstance(event, TextChunk):
            if reasoning_printed:
                print(f"{RESET}\n{model_name}:    {BOLD}", end="", flush=True)
                reasoning_printed = False
            print(event.content, end="", flush=True)

        elif isinstance(event, ToolCall):
            print(f"\n{GREEN}[调用工具] {event.name}{RESET}")

        elif isinstance(event, ToolResult):
            print(f"{GREEN}[工具返回] {event.name}: {event.result}{RESET}")

        elif isinstance(event, AgentComplete):
            print(RESET)
            print("-" * 50)
            print("任务圆满完成！")


if __name__ == "__main__":
    main()
