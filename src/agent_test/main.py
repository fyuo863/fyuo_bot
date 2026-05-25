from core.chat import AgentChat, TextChunk, ToolCall
from tools.base import GetWeatherTool, GetLocationTool
from core.usage import Usage
import json

def main():
    # 1. 准备工具 (去掉了 GetEndTool，因为不需要)
    weather_tool = GetWeatherTool()
    location_tool = GetLocationTool()
    tools = [weather_tool.to_openai_schema(), location_tool.to_openai_schema()]

    input_text = "我这里今天天气怎么样？"

    print("=" * 50)
    print(f"User:  {input_text}")
    print("=" * 50)

    # 2. 初始化全局对话上下文（标准的 OpenAI 格式）
    messages = [
        {"role": "system", "content": "你是一个生活助手，可以调用工具获取信息。"},
        {"role": "user", "content": input_text}
    ]

    # 3. 开启标准的 ReAct 核心执行循环
    while True:
        stream_items = []
        print("AI:    ", end="", flush=True)

        # 【核心修改】直接把整个 messages 列表传给模型，而不是传拼接的字符串
        for item in AgentChat.chat(messages=messages, tools=tools):
            stream_items.append(item)
            
            if isinstance(item, TextChunk):
                print(item.content, end="", flush=True)
            elif isinstance(item, ToolCall):
                print(f"\n[思考中...准备调用工具] {item.name}")
        print() # 换行

        # 4. 本地拼装本次对话产生的碎片，并将其作为记忆追加到 messages 中
        assistant_msg = Usage.assemble_assistant_message(stream_items)
        messages.append(assistant_msg)

        # 5. 【判断终止条件】如果本次回复没有包含工具调用，说明大模型已经得出了最终答案
        if "tool_calls" not in assistant_msg:
            print("-" * 50)
            print("任务圆满完成！")
            break

        # 6. 如果代码走到这里，说明有工具需要被执行
        for tool_call in assistant_msg["tool_calls"]:
            func_name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])
            
            # 路由并执行本地工具
            result = ""
            if func_name == weather_tool.name:
                result = weather_tool.execute(**args)
            elif func_name == location_tool.name:
                result = location_tool.execute(**args)
            
            print(f"[本地执行完成] {func_name} 返回结果: {result}")
            
            # 7. 【回传结果】将真实的执行结果打包成 tool 角色，追加到上下文中
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": str(result)
            })
            
        # 循环继续，带着装满了工具返回结果的 messages 再次请求大模型...
        print("-" * 50)

if __name__ == "__main__":
    main()