"""
MemoryManager —— 管理 .fyuobot/memories/ 下的持久记忆文件。

- MEMORY.md: 客观事实、项目规范、工作流习惯、教训 (上限 2,200 字符)
- USER.md: 用户身份、沟通偏好、技术水平 (上限 1,375 字符)

每次会话启动时生成“冻结快照”注入系统提示词，会话期间不变。
Agent 通过 add_memory / remove_memory 工具修改记忆，
超出容量时触发 LLM 自动整合（Consolidation）。
"""

import os


class MemoryManager:

    MEMORY_DIR = ".fyuobot/memories"
    LIMITS = {"MEMORY": 2200, "USER": 1375}

    def __init__(self, workspace: str):
        self.workspace = workspace
        self.memory_dir = os.path.join(workspace, self.MEMORY_DIR)
        os.makedirs(self.memory_dir, exist_ok=True)
        # 确保两个文件存在
        self._ensure_file("MEMORY")
        self._ensure_file("USER")

    # ---------------- 文件操作 ----------------

    def _file_path(self, file_type: str) -> str:
        return os.path.join(self.memory_dir, f"{file_type}.md")

    def _ensure_file(self, file_type: str):
        path = self._file_path(file_type)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("")

    def _read(self, file_type: str) -> str:
        with open(self._file_path(file_type), "r", encoding="utf-8") as f:
            return f.read()

    def _write(self, file_type: str, content: str):
        with open(self._file_path(file_type), "w", encoding="utf-8") as f:
            f.write(content)

    # ---------------- 快照 ----------------

    def get_snapshot(self) -> str:
        """返回冻结快照，用于注入系统提示词。"""
        memory = self._read("MEMORY").strip()
        user = self._read("USER").strip()
        parts = []
        if memory:
            parts.append(f"## 项目记忆\n{memory}")
        if user:
            parts.append(f"## 用户信息\n{user}")
        return "\n\n".join(parts)

    # ---------------- 增 / 删 / 查 ----------------

    def add(self, file_type: str, text: str) -> str:
        """添加记忆条目。超出容量时报错并返回当前全部内容。"""
        if file_type not in self.LIMITS:
            return f"错误：未知记忆类型 '{file_type}'，可选 MEMORY 或 USER。"
        limit = self.LIMITS[file_type]
        current = self._read(file_type).strip()
        # 构造新内容
        new_content = (current + "\n" + text).strip() if current else text
        if len(new_content) > limit:
            return (
                f"容量不足：添加后 {file_type} 将达 {len(new_content)} 字符，"
                f"超出上限 {limit} 字符（超出 {len(new_content) - limit}）。\n\n"
                f"当前 {file_type} 全文如下，请用 remove_memory 整合精简后再添加：\n"
                f"---\n{current}\n---"
            )
        self._write(file_type, new_content)
        return f"已添加到 {file_type}（当前 {len(new_content)}/{limit} 字符）。"

    def remove(self, file_type: str, old_text: str, new_text: str | None = None) -> str:
        """通过唯一子串匹配定位并删除/替换。"""
        if file_type not in self.LIMITS:
            return f"错误：未知记忆类型 '{file_type}'，可选 MEMORY 或 USER。"
        content = self._read(file_type)
        count = content.count(old_text)
        if count == 0:
            return f"错误：在 {file_type} 中未找到匹配 '{old_text}' 的内容。"
        if count > 1:
            return (
                f"错误：'{old_text}' 在 {file_type} 中匹配了 {count} 处，"
                "请提供更长的唯一子串以精确定位。"
            )
        if new_text is None:
            new_content = content.replace(old_text, "")
        else:
            new_content = content.replace(old_text, new_text)
        self._write(file_type, new_content)
        action = "替换" if new_text is not None else "删除"
        return f"已{action} {file_type} 中的匹配条目（当前 {len(new_content.strip())}/{self.LIMITS[file_type]} 字符）。"

    def get_all(self, file_type: str) -> str:
        """获取当前全部记忆内容。"""
        if file_type not in self.LIMITS:
            return f"错误：未知记忆类型 '{file_type}'。"
        content = self._read(file_type).strip()
        return content if content else f"{file_type} 当前为空。"
