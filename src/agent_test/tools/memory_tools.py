"""
记忆持久化工具 —— 供 Agent 调用的 add_memory / remove_memory。

- add_memory: 添加新记忆，超出容量时返回错误及当前全文，触发 Agent 整合。
- remove_memory: 通过唯一子串匹配定位，删除或替换旧记忆。
"""

from dataclasses import dataclass, field

from .base import BaseTool


@dataclass
class AddMemoryTool(BaseTool):

    name: str = "add_memory"
    description: str = (
        "添加一条新的持久记忆。记忆分为两种类型："
        "MEMORY（项目规范、工作习惯、经验教训等客观事实）和 "
        "USER（用户身份、偏好、技术水平等个性化信息）。"
        "记录高信息密度的精简短句。"
        "如果返回容量不足的错误，请用 get_memory 查看现有内容，"
        "用 remove_memory 整合精简后再重试。"
    )
    parameters: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["MEMORY", "USER"],
                "description": "记忆类型：MEMORY=客观事实/项目规范，USER=用户个人信息",
            },
            "content": {
                "type": "string",
                "description": "要添加的记忆内容，建议为一句高密度精简短句",
            },
        },
        "required": ["type", "content"],
    })

    memory_manager: object = None

    def execute(self, type: str = "", content: str = "", **kwargs) -> str:
        if not self.memory_manager:
            return "错误：MemoryManager 未初始化。"
        return self.memory_manager.add(type, content)


@dataclass
class RemoveMemoryTool(BaseTool):

    name: str = "remove_memory"
    description: str = (
        "通过唯一子串匹配删除或替换旧记忆。只需提供原记忆中独一无二的"
        "一小段文字（old_text）即可精准定位。若提供 new_text 则替换，"
        "否则直接删除该条目。用于修正过时信息或整合精简记忆。"
    )
    parameters: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["MEMORY", "USER"],
                "description": "记忆类型：MEMORY 或 USER",
            },
            "old_text": {
                "type": "string",
                "description": "原记忆中独一无二的一小段文字，用于定位要删除/替换的条目",
            },
            "new_text": {
                "type": "string",
                "description": "替换后的新文本。留空则直接删除匹配条目。",
            },
        },
        "required": ["type", "old_text"],
    })

    memory_manager: object = None

    def execute(self, type: str = "", old_text: str = "", new_text: str = "", **kwargs) -> str:
        if not self.memory_manager:
            return "错误：MemoryManager 未初始化。"
        new = new_text if new_text else None
        return self.memory_manager.remove(type, old_text, new)


@dataclass
class GetMemoryTool(BaseTool):

    name: str = "get_memory"
    description: str = (
        "查看当前全部记忆内容。在整合精简记忆前使用，"
        "以便了解现有条目并做出合并/删除决策。"
    )
    parameters: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["MEMORY", "USER"],
                "description": "要查看的记忆类型",
            },
        },
        "required": ["type"],
    })

    memory_manager: object = None

    def execute(self, type: str = "", **kwargs) -> str:
        if not self.memory_manager:
            return "错误：MemoryManager 未初始化。"
        return self.memory_manager.get_all(type)
