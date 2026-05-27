"""
HistoryManager —— SQLite 持久化对话历史，支持 LLM 自动浓缩与话题分类。

- save_turn 调用 LLM 判断是否值得保存、按话题分类、浓缩对话
- 通过 LIKE 模糊匹配搜索历史（搜索 summary 和 topic）
- 数据库文件: .fyuobot/history.db
"""

import json
import os
import sqlite3
import time

from openai import OpenAI
from config.config import api_key, base_url, model_name


class HistoryManager:

    DB_FILENAME = "history.db"

    def __init__(self, workspace: str):
        db_dir = os.path.join(workspace, ".fyuobot")
        os.makedirs(db_dir, exist_ok=True)
        self.db_path = os.path.join(db_dir, self.DB_FILENAME)
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            # 检测旧表结构并迁移
            cursor = conn.execute("PRAGMA table_info(conversations)")
            columns = {row[1] for row in cursor.fetchall()}
            if columns and "role" in columns:
                conn.execute("DROP TABLE conversations")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    topic TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL
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
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conv_topic
                ON conversations(topic)
            """)

    # ---------------- LLM 浓缩与分类 ----------------

    CONDENSE_PROMPT = (
        "你是一个对话摘要器。分析以下一轮用户与AI助手的对话，完成三个任务：\n\n"
        "1. **判断是否值得保存**：如果对话内容只是简单寒暄、无信息量的确认、"
        "或者纯闲聊，则标记为 SKIP。有实质信息（知识、决策、代码、问题解决、"
        "用户偏好、项目信息等）的对话才值得保存。\n\n"
        "2. **话题分类**：用 2-5 个字的简短标签对话题分类，例如：\n"
        "   Python编程、代码审查、Bug修复、配置管理、\n"
        "   项目规划、技术选型、调试日志、记忆管理\n\n"
        "3. **浓缩对话**：用 1-3 句中文精炼这段对话的核心信息。\n"
        "   保留关键细节（文件名、函数名、具体数据、决策理由），删除废话。\n"
        "   **重要：summary 中所有双引号必须用中文引号「」替代，严禁出现英文双引号\"。**\n\n"
        "请严格按照以下 JSON 格式返回，不要加任何其他文字：\n"
        "{{\"action\": \"SKIP\"}}  或  {{\"action\": \"SAVE\", \"topic\": \"话题标签\", \"summary\": \"浓缩摘要\"}}\n\n"
        "=== 对话内容 ===\n"
        "用户: {user_input}\n"
        "助手: {agent_response}\n"
    )

    def _condense_turn(self, user_input: str, agent_response: str) -> dict | None:
        """调用 LLM 判断是否保存、分类话题、浓缩对话。

        Returns:
            None 表示 LLM 判断不需要保存。
            dict 包含 topic 和 summary。
        """
        prompt = self.CONDENSE_PROMPT.format(
            user_input=user_input[:3000],
            agent_response=agent_response[:3000],
        )
        try:
            response = self._client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                stream=False,
            )
            text = response.choices[0].message.content.strip()
            # 清理可能的 markdown 代码块包裹
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            data = json.loads(text)
            if data.get("action") == "SAVE" and data.get("summary"):
                return {"topic": data.get("topic", ""), "summary": data["summary"]}
            return None
        except json.JSONDecodeError:
            # JSON 解析失败，尝试正则兜底（summary 中可能有未转义的引号）
            return self._parse_fallback(text)
        except Exception as e:
            print(f"[历史] 浓缩失败: {e}")
            return None

    def _parse_fallback(self, text: str) -> dict | None:
        """正则兜底：当 JSON 解析失败时，直接从文本中提取字段。"""
        import re
        action_match = re.search(r'"action"\s*:\s*"(SKIP|SAVE)"', text)
        if not action_match or action_match.group(1) != "SAVE":
            return None

        topic = ""
        topic_match = re.search(r'"topic"\s*:\s*"([^"]*)"', text)
        if topic_match:
            topic = topic_match.group(1)

        # summary 可能包含未转义引号，取 "summary": " 之后到末尾 "} 之前
        summary = ""
        summary_start = re.search(r'"summary"\s*:\s*"', text)
        if summary_start:
            rest = text[summary_start.end():]
            # 从后往前找最后一个 "}（JSON 对象的闭合），避免 summary 内含 "} 时截错
            last_close = rest.rstrip().rfind('"}')
            if last_close >= 0:
                summary = rest[:last_close]
            else:
                summary = rest.strip().rstrip('"').rstrip("}").strip('"').strip()

        if summary:
            return {"topic": topic, "summary": summary}
        return None

    # ---------------- 保存 ----------------

    def save_turn(self, session_id: str, user_input: str, agent_response: str):
        """保存一轮对话：经 LLM 判断 + 话题分类 + 浓缩后存入。"""
        condensed = self._condense_turn(user_input, agent_response)
        if condensed is None:
            return  # LLM 判断不值得保存

        now = time.time()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO conversations (session_id, timestamp, topic, summary) "
                "VALUES (?, ?, ?, ?)",
                (session_id, now, condensed["topic"], condensed["summary"]),
            )

    # ---------------- 搜索 ----------------

    def search(self, query: str, limit: int = 5) -> str:
        """关键词搜索历史（搜索 summary 和 topic）。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT session_id, timestamp, topic, summary FROM conversations "
                "WHERE summary LIKE ? OR topic LIKE ? "
                "ORDER BY timestamp DESC "
                "LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()

        if not rows:
            return f"未找到与 '{query}' 相关的历史记录。"

        lines = [f"搜索 '{query}' 找到 {len(rows)} 条记录："]
        for sid, ts, topic, summary in rows:
            time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
            topic_str = f" [{topic}]" if topic else ""
            lines.append(f"[{time_str}]{topic_str} ({sid}): {summary}")
        return "\n".join(lines)

    # ---------------- 最近记录 ----------------

    def get_recent(self, limit: int = 10) -> str:
        """获取最近的对话记录。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT session_id, timestamp, topic, summary FROM conversations "
                "ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()

        if not rows:
            return "暂无历史记录。"

        lines = [f"最近 {len(rows)} 条记录："]
        for sid, ts, topic, summary in reversed(rows):
            time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
            topic_str = f" [{topic}]" if topic else ""
            lines.append(f"[{time_str}]{topic_str}: {summary}")
        return "\n".join(lines)

    def get_recent_history(self, limit: int = 10) -> str:
        """获取最近历史记录（兼容旧接口）。"""
        return self.get_recent(limit=limit)
