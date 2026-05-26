"""
MemoryManager —— 管理 .fyuobot/memories/ 下的持久记忆文件。

- MEMORY.md: 客观事实、项目规范、工作流习惯、教训 (上限 2,200 字符)
- USER.md: 用户身份、沟通偏好、技术水平 (上限 1,375 字符)

每次会话启动时生成"冻结快照"注入系统提示词，会话期间不变。
Agent 通过 replace_memory 工具修改记忆（统一追加/替换/删除），
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

    # ---------------- 统一替换接口 ----------------

    def replace(self, file_type: str, old_text: str, new_text: str) -> str:
        """统一的记忆修改接口。

        - old_text 为空 → 追加 new_text（新增模式）
        - old_text 精确匹配一处 → 替换为 new_text（替换/删除模式）
        - 始终检查容量限制，超出时返回错误 + 当前全文
        """
        if file_type not in self.LIMITS:
            return f"错误：未知记忆类型 '{file_type}'，可选 MEMORY 或 USER。"

        limit = self.LIMITS[file_type]
        current = self._read(file_type)

        # ---- 新增模式 ----
        if not old_text.strip():
            new_content = (current.rstrip() + "\n" + new_text).strip() if current.strip() else new_text
            if len(new_content) > limit:
                return self._capacity_error(file_type, current, new_content, limit)
            self._write(file_type, new_content)
            return f"已追加到 {file_type}（{len(new_content)}/{limit} 字符）。"

        # ---- 替换模式 ----
        count = current.count(old_text)
        if count == 0:
            return (
                f"错误：在 {file_type} 中未找到匹配 '{old_text}' 的内容。\n"
                f"如果要新增，请将 old_text 留空。\n"
                f"当前 {file_type} 全文：\n---\n{current.strip() or '(空)'}\n---"
            )
        if count > 1:
            return (
                f"错误：'{old_text}' 在 {file_type} 中匹配了 {count} 处，"
                "请提供更长的唯一子串以精确定位。"
            )

        new_content = current.replace(old_text, new_text)
        stripped = new_content.strip()
        if len(stripped) > limit:
            return self._capacity_error(file_type, new_content if current.strip() else stripped, stripped, limit)

        self._write(file_type, new_content)
        if new_text:
            return f"已替换 {file_type} 中的匹配条目（{len(stripped)}/{limit} 字符）。"
        else:
            return f"已删除 {file_type} 中的匹配条目（{len(stripped)}/{limit} 字符）。"

    def _capacity_error(self, file_type: str, current: str, new_content: str, limit: int) -> str:
        return (
            f"容量不足：操作后 {file_type} 将达 {len(new_content.strip())} 字符，"
            f"超出上限 {limit} 字符（超出 {len(new_content.strip()) - limit}）。\n\n"
            f"当前 {file_type} 全文如下，请整合精简后重试：\n"
            f"---\n{current.strip()}\n---"
        )

    def get_all(self, file_type: str) -> str:
        """获取当前全部记忆内容。"""
        if file_type not in self.LIMITS:
            return f"错误：未知记忆类型 '{file_type}'。"
        content = self._read(file_type).strip()
        return content if content else f"{file_type} 当前为空。"
