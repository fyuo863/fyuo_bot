"""
HistoryManager —— SQLite 持久化所有历史对话，支持关键词搜索。

- 每次对话轮次保存 user 输入 + assistant 输出
- 通过 LIKE 模糊匹配搜索历史
- 数据库文件: .fyuobot/history.db
"""

import os
import sqlite3
import time


class HistoryManager:

    DB_FILENAME = "history.db"

    def __init__(self, workspace: str):
        db_dir = os.path.join(workspace, ".fyuobot")
        os.makedirs(db_dir, exist_ok=True)
        self.db_path = os.path.join(db_dir, self.DB_FILENAME)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conv_session
                ON conversations(session_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conv_timestamp
                ON conversations(timestamp)
            """)

    def save_turn(self, session_id: str, user_input: str, agent_response: str):
        """保存一轮对话（user + assistant 各一条）。"""
        now = time.time()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO conversations (session_id, timestamp, role, content) VALUES (?, ?, ?, ?)",
                (session_id, now, "user", user_input),
            )
            conn.execute(
                "INSERT INTO conversations (session_id, timestamp, role, content) VALUES (?, ?, ?, ?)",
                (session_id, now, "assistant", agent_response),
            )

    def search(self, query: str, limit: int = 5) -> str:
        """关键词搜索历史，返回格式化的结果。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT session_id, timestamp, role, content FROM conversations "
                "WHERE content LIKE ? "
                "ORDER BY timestamp DESC "
                "LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()

        if not rows:
            return f"未找到与 '{query}' 相关的历史记录。"

        lines = [f"搜索 '{query}' 找到 {len(rows)} 条记录："]
        for sid, ts, role, content in rows:
            time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
            tag = "User" if role == "user" else "Assistant"
            snippet = content[:300] + ("..." if len(content) > 300 else "")
            lines.append(f"[{time_str}] {tag}({sid}): {snippet}")
        return "\n".join(lines)

    def get_recent(self, limit: int = 10) -> str:
        """获取最近的对话记录。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT session_id, timestamp, role, content FROM conversations "
                "ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()

        if not rows:
            return "暂无历史记录。"

        lines = [f"最近 {len(rows)} 条记录："]
        for sid, ts, role, content in reversed(rows):
            time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
            tag = "User" if role == "user" else "Assistant"
            snippet = content[:300] + ("..." if len(content) > 300 else "")
            lines.append(f"[{time_str}] {tag}({sid}): {snippet}")
        return "\n".join(lines)

    # --- 新增：获取最近的历史记录 ---
    def get_recent_history(self, limit: int = 10) -> str:
        """
        不进行文本匹配，直接拉取最近的 N 条对话记录。
        用于大模型进行全局总结或回忆最近上下文。
        """
        try:
            # 这里的 SQL 语句需要根据您的实际表结构调整
            # 假设表名叫 conversations，字段有 role, content, timestamp
            # 我们按照时间倒序(DESC)拉取最新记录，但在返回前最好正序排列(ASC)以便阅读
            self.cursor.execute(
                """
                SELECT role, content 
                FROM (
                    SELECT role, content, timestamp 
                    FROM conversations 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ) 
                ORDER BY timestamp ASC
                """,
                (limit,)
            )
            rows = self.cursor.fetchall()
            
            if not rows:
                return "未找到任何最近的历史记录。"

            # 将结果组装成纯文本供大模型阅读
            result_lines = ["[最近的历史记录]"]
            for row in rows:
                role, content = row
                result_lines.append(f"{role.capitalize()}: {content}")
            
            return "\n".join(result_lines)
            
        except sqlite3.Error as e:
            return f"查询数据库时发生错误: {str(e)}"
