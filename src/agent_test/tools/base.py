"""
工具模板 —— 复制此文件来创建新工具。

每个工具需要提供：
1. name / description   — 给大模型看的，决定何时调用
2. parameters           — JSON Schema 格式，定义参数
3. to_openai_schema()   — 转成 OpenAI tool 定义（基类已实现）
4. execute(**kwargs)    — 实际执行逻辑
"""
import os
from abc import ABC, abstractmethod
import requests
import asyncio


class BaseTool(ABC):
    """工具基类。workspace 由 Agent 启动时自动注入。"""

    name: str = ""
    description: str = ""
    parameters: dict = {}
    workspace: str = ""

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def resolve_path(self, relative_path: str) -> str:
        """安全解析工作区内路径，阻止 ../ 越权。返回绝对路径。"""
        if not self.workspace:
            raise RuntimeError("workspace 未设置")
        ws = os.path.abspath(self.workspace)
        target = os.path.normpath(os.path.join(ws, relative_path))
        target = os.path.abspath(target)
        if os.path.commonpath([ws, target]) != ws:
            raise ValueError(f"禁止访问工作区外的路径: {relative_path}")
        return target

    @abstractmethod
    def execute(self, **kwargs) -> str:
        ...


# ============================================================
# 内置工具
# ============================================================

class GetWeatherTool(BaseTool):
    name = "get_weather"
    description = "获取指定城市的当前天气"
    parameters = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称，例如 Beijing"},
        },
        "required": ["city"],
    }

    def execute(self, city: str = "", **kwargs) -> str:
        return f"{city} 当前天气：晴，25°C，湿度 60%"


class GetLocationTool(BaseTool):
    name = "get_location"
    description = "获取用户当前所在的城市名称"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "固定传 'self'"},
        },
        "required": [],
    }

    def execute(self, **kwargs) -> str:
        try:
            response = requests.get("http://ip-api.com/json/", timeout=5)
            data = response.json()
            if data["status"] == "success":
                return data["city"]
            return "未知城市"
        except Exception as e:
            return f"定位失败: {str(e)}"


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "读取工作区内指定路径的文件内容"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "相对于工作区的文件路径，例如 'src/main.py'",
            }
        },
        "required": ["file_path"],
    }

    def execute(self, file_path: str = "", **kwargs) -> str:
        try:
            abs_path = self.resolve_path(file_path)
        except (ValueError, RuntimeError) as e:
            return str(e)
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"读取文件失败: {str(e)}"


class ListFilesTool(BaseTool):
    name = "list_files"
    description = "列出工作区内指定目录的文件和子目录"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "相对于工作区的目录路径，留空表示根目录",
            }
        },
        "required": [],
    }

    def execute(self, path: str = "", **kwargs) -> str:
        try:
            abs_path = self.resolve_path(path) if path else os.path.abspath(self.workspace)
        except (ValueError, RuntimeError) as e:
            return str(e)
        try:
            entries = os.listdir(abs_path)
            lines = [f"{abs_path}/"]
            for name in sorted(entries):
                full = os.path.join(abs_path, name)
                tag = "/" if os.path.isdir(full) else ""
                lines.append(f"  {name}{tag}")
            return "\n".join(lines)
        except Exception as e:
            return f"列出目录失败: {str(e)}"


class LetUserAnswer(BaseTool):
    name = "let_user_answer"
    description = (
        "当你需要用户提供额外信息或确认时，必须使用此工具。"
        "切勿在文本回复中直接提问，必须通过此工具。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "需要向用户提出的具体问题或要求",
            }
        },
        "required": ["question"],
    }

    def execute(self, question: str = "", **kwargs) -> str:
        return f"USER_INPUT_REQUIRED: {question}"


class MCPToolAdapter(BaseTool):
    """通用适配器，将远端 MCP Server 工具包装为 BaseTool。"""

    def __init__(self, mcp_client, tool_name: str, description: str, parameters: dict):
        self.mcp_client = mcp_client
        self.name = tool_name
        self.description = description
        self.parameters = parameters

    def execute(self, **kwargs) -> str:
        try:
            result = asyncio.run(self._async_execute(**kwargs))
            return str(result)
        except Exception as e:
            return f"MCP 工具 {self.name} 错误: {str(e)}"

    async def _async_execute(self, **kwargs):
        result = await self.mcp_client.call_tool(self.name, arguments=kwargs)
        return result.content
