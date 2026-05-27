from core.chat import AgentChat, TextChunk, ToolCall


class Usage:
    @staticmethod
    def assemble_assistant_message(stream_items: list) -> dict:
        """
        将流式输出的碎片拼装成标准的 assistant 消息字典，支持深度思考模型
        """
        content = ""
        reasoning_content = ""  # 【新增】用于收集思考过程
        tool_calls = []

        for item in stream_items:
            if isinstance(item, TextChunk):
                content += item.content
            # 【新增】如果是思考片段，拼接到思考字符串里
            elif type(item).__name__ == "ReasoningChunk": 
                reasoning_content += item.content
            elif isinstance(item, ToolCall):
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
        
        # 组装普通文本
        if content:
            message["content"] = content
        else:
            # 某些模型可能只有 tool_calls 没有 content，为防止严格校验报错，给个空字符串
            message["content"] = ""
            
        # 【关键修复】如果模型输出了思考过程，必须原样塞回给它！
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
            
        # 组装工具调用
        if tool_calls:
            message["tool_calls"] = tool_calls

        return message