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

class NewFileTool(BaseTool):
    name = "new_file"
    description = (
        "在工作区内创建新文件或文件夹。创建完成后自动检查是否成功。"
        "如果是文件，会创建空文件；如果是文件夹，会递归创建所有父目录。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "相对于工作区的路径，例如 'src/utils/helper.py' 或 'data/output/'",
            },
            "type": {
                "type": "string",
                "enum": ["file", "folder"],
                "description": "创建类型：file（文件）或 folder（文件夹）",
            },
        },
        "required": ["path", "type"],
    }

    def execute(self, path: str = "", type: str = "file", **kwargs) -> str:
        try:
            abs_path = self.resolve_path(path)
        except (ValueError, RuntimeError) as e:
            return str(e)

        try:
            if type == "folder":
                os.makedirs(abs_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                with open(abs_path, "w", encoding="utf-8") as f:
                    pass

            # 自检：确认创建成功
            exists = os.path.isdir(abs_path) if type == "folder" else os.path.isfile(abs_path)
            if exists:
                kind = "文件夹" if type == "folder" else "文件"
                return f"创建{kind}成功: {path}"
            return f"创建失败：无法验证 {path} 是否存在"
        except Exception as e:
            return f"创建失败: {str(e)}"


class WriteFileTool(BaseTool):
    name = "write_file"
    description = (
        "将内容写入工作区内的指定文件。使用 Python 原生文件流写入，"
        "如果文件不存在会自动创建，如果已存在则覆盖。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "相对于工作区的文件路径，例如 'src/main.py'",
            },
            "content": {
                "type": "string",
                "description": "要写入文件的完整文本内容",
            },
        },
        "required": ["file_path", "content"],
    }

    def execute(self, file_path: str = "", content: str = "", **kwargs) -> str:
        try:
            abs_path = self.resolve_path(file_path)
        except (ValueError, RuntimeError) as e:
            return str(e)

        try:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)

            # 自检：读回确认写入
            with open(abs_path, "r", encoding="utf-8") as f:
                written = f.read()
            if written == content:
                lines = content.count("\n") + 1
                size = len(content)
                return f"写入完成: {file_path}（{lines} 行，{size} 字符）"
            return f"写入异常：内容校验不匹配（写入 {len(content)} 字符，读回 {len(written)} 字符）"
        except Exception as e:
            return f"写入失败: {str(e)}"


class DoCommand(BaseTool):
    name = "do_command"
    description = (
        "在工作区内执行 shell 命令。执行前会请求用户确认。"
        "用于运行脚本、编译代码、安装依赖、git 操作等。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 shell 命令，例如 'python main.py' 或 'ls src/'",
            }
        },
        "required": ["command"],
    }

    def execute(self, command: str = "", **kwargs) -> str:
        import subprocess

        # 请求用户确认
        print(f"\n\033[33m[命令审批]\033[0m 即将执行:")
        print(f"  \033[1m{command}\033[0m")
        approval = input("是否同意执行? (y/n): ").strip().lower()
        if approval != "y":
            return f"用户拒绝了命令: {command}"

        print(f"\033[2m执行中...\033[0m")
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace or ".",
                capture_output=True,
                text=True,
                timeout=30,
            )
            out = result.stdout
            if result.stderr:
                out += f"\n[stderr]\n{result.stderr}"
            if result.returncode != 0:
                out += f"\n(返回码: {result.returncode})"
            return out or "(无输出)"
        except subprocess.TimeoutExpired:
            return "命令执行超时 (30s)"
        except Exception as e:
            return f"命令执行失败: {str(e)}"
