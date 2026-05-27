"""
记忆与历史工具 —— 供 Agent 调用的 replace_memory / get_history。

- replace_memory: 统一的记忆修改（追加 / 替换 / 删除），基于唯一子串匹配。
- get_history: 从 SQLite 搜索历史对话。
"""

from dataclasses import dataclass, field

from .base import BaseTool


@dataclass
class ReplaceMemoryTool(BaseTool):

    name: str = "replace_memory"
    description: str = (
        "修改持久记忆。记忆分为 MEMORY（项目规范、工作习惯、经验教训）"
        "和 USER（用户身份、偏好、技术水平）。\n"
        "三种模式：\n"
        "  - 新增：old_text 留空，new_text 填要添加的内容\n"
        "  - 替换：old_text 填原记忆中的唯一子串，new_text 填替换内容\n"
        "  - 删除：old_text 填要删除的唯一子串，new_text 留空\n"
        "超出容量时会返回错误及当前全文，需整合精简后重试。"
    )
    parameters: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["MEMORY", "USER"],
                "description": "记忆类型",
            },
            "old_text": {
                "type": "string",
                "description": "原记忆中要替换/删除的片段（唯一子串），留空表示追加新条目",
            },
            "new_text": {
                "type": "string",
                "description": "替换后的新文本。old_text 留空时表示要添加的内容；new_text 留空时表示删除",
            },
        },
        "required": ["type", "old_text", "new_text"],
    })

    memory_manager: object = None

    def execute(self, type: str = "", old_text: str = "", new_text: str = "", **kwargs) -> str:
        if not self.memory_manager:
            return "错误：MemoryManager 未初始化。"
        return self.memory_manager.replace(type, old_text, new_text)


@dataclass
class GetHistoryTool(BaseTool):
    name: str = "get_history"
    # 【修改 1】优化描述：明确告诉大模型什么情况下传空，什么情况下传词，且教它避开废话
    description: str = (
        "获取历史对话记录（已自动浓缩并分类）。\n"
        "1. 如果用户要求'总结之前的聊天'或没有特定的搜索目标，请将 query 留空，以获取最近的历史记录。\n"
        "2. 如果用户寻找特定话题，请提取【话题分类标签】或【具体的核心名词】（如'Python编程'、'代码审查'、'Bug修复'、'配置'）。\n"
        "警告：绝不能使用'聊天'、'刚才'、'用户'等无意义的口语词作为关键词！"
    )
    parameters: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词。如果不需要特定关键词，请务必传入空字符串 \"\"",
            },
            # 【修改 2】增加 limit 参数，让大模型可以控制召回范围
            "limit": {
                "type": "integer",
                "description": "需要返回的最近对话条数，默认 10，最多 50",
            }
        },
        # 【修改 3】query 依然可以必填，但允许大模型传 "" (空字符串)
        "required": ["query"], 
    })

    history_manager: object = None

    def execute(self, query: str = "", limit: int = 10, **kwargs) -> str:
        if not self.history_manager:
            return "错误：HistoryManager 未初始化。"
        
        # 逻辑分流
        if not query or query.strip() == "":
            # 注意这里方法名叫 get_recent
            return self.history_manager.get_recent(limit=limit) 
        else:
            return self.history_manager.search(query, limit=limit)