"""
工具模板 —— 复制此文件来创建新工具。

每个工具需要提供：
1. name / description   — 给大模型看的，决定何时调用
2. parameters           — JSON Schema 格式，定义参数
3. to_openai_schema()   — 转成 OpenAI tool 定义（基类已实现）
4. execute(**kwargs)    — 实际执行逻辑

快速创建步骤：
  1. 复制 GetWeatherTool，改类名
  2. 修改 name / description / parameters 三个类属性
  3. 实现 execute 方法
"""
from abc import ABC, abstractmethod
import requests


class BaseTool(ABC):
    """工具基类 —— 所有工具继承这个，覆盖三个类属性即可"""

    name: str = ""
    description: str = ""
    parameters: dict = {}

    def to_openai_schema(self) -> dict:
        """生成 OpenAI 兼容的 tool 定义"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具逻辑，返回字符串结果"""
        ...


# ============================================================
# 示例工具
# ============================================================

class GetWeatherTool(BaseTool):
    name = "get_weather"
    description = "获取指定城市的当前天气"
    parameters = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称，例如 Beijing",
            }
        },
        "required": ["city"],
    }

    def execute(self, city: str = "", **kwargs) -> str:
        # 这里接入真实天气 API
        return f"{city} 当前天气：晴，25°C，湿度 60%"


class GetLocationTool(BaseTool):
    name = "get_location"
    description = "获取用户当前所在的城市名称"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "固定传 'self'"}
        },
        "required": []
    }

    def execute(self, **kwargs) -> str:
        try:
            # 调用免费 IP 定位服务
            response = requests.get("http://ip-api.com/json/", timeout=5)
            data = response.json()
            if data['status'] == 'success':
                return data['city'] # 返回 "Hangzhou" 等
            return "未知城市"
        except Exception as e:
            return f"定位失败: {str(e)}"

class ReadFileTool(BaseTool):
    name = "read_file"
    description = "读取指定路径的文件内容"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件的完整路径，例如 'workspace/data.txt'"
            }
        },
        "required": ["file_path"]
    }

    def execute(self, file_path: str, **kwargs) -> str:
        # 安全检查：限制只能读取特定目录，防止越权读取系统文件
        if not file_path.startswith("workspace/"):
            return "错误：禁止访问该路径的文件。"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"读取文件失败: {str(e)}"