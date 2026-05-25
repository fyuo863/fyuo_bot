from core.chat import AgentChat, TextChunk, ToolCall


class Usage:
    def assemble_assistant_message(stream_items: list) -> dict:
        """
        将流式输出的碎片拼装成标准的 assistant 消息字典
        """
        content = ""
        tool_calls = []

        for item in stream_items:
            if isinstance(item, TextChunk):
                content += item.content
            elif isinstance(item, ToolCall):
                # 将自定义的 ToolCall 对象转换为 OpenAI 标准格式
                tool_calls.append({
                    "id": item.call_id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": item.arguments
                    }
                })

        # 构建基础的 assistant 消息
        message = {"role": "assistant"}
        
        # 只有当 content 不为空时才添加，保持整洁
        if content:
            message["content"] = content
            
        # 只有当有工具调用时才添加 tool_calls 字段
        if tool_calls:
            message["tool_calls"] = tool_calls

        return message