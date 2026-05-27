"""
HistoryManager —— 双层对话历史存储。

- save_turn 被动追加原始对话到 .fyuobot/memories/HISTORY.md（agent 不可见）
- 当 HISTORY.md 超过阈值时，自动调用 LLM 批量浓缩存入 SQLite
- get_history 搜索 SQLite 浓缩历史
"""

import json
import os
import re
import sqlite3
import time
import datetime
import threading

from openai import OpenAI
from config.config import api_key, base_url, model_name


class HistoryManager:

    DB_FILENAME = "history.db"
    HISTORY_REL_PATH = ".fyuobot/memories/HISTORY.md"
    MAX_BUFFER_CHARS = 15000          # HISTORY.md 超过此值触发浓缩
    KEEP_RECENT_CHARS = 3000          # 浓缩后保留最近的原始对话

    def __init__(self, workspace: str):
        self._workspace = workspace
        db_dir = os.path.join(workspace, ".fyuobot")
        os.makedirs(db_dir, exist_ok=True)
        self.db_path = os.path.join(db_dir, self.DB_FILENAME)
        self._history_path = os.path.join(workspace, self.HISTORY_REL_PATH)
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._condense_lock = threading.Lock()
        self._init_db()
        self._start_session()

    # ==================== SQLite ====================

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
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
            for idx in ("session", "timestamp", "topic"):
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_conv_{idx} "
                    f"ON conversations({idx})"
                )

    def _insert_condensed(self, entries: list[dict]):
        """将浓缩条目批量写入 SQLite。"""
        now = time.time()
        with self._get_conn() as conn:
            for entry in entries:
                conn.execute(
                    "INSERT INTO conversations (session_id, timestamp, topic, summary) "
                    "VALUES (?, ?, ?, ?)",
                    (entry.get("session", ""), now, entry.get("topic", ""), entry["summary"]),
                )

    # ==================== HISTORY.md 缓冲 ====================

    def _start_session(self):
        """每次程序启动开启一个新会话。"""
        self._session_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        # 统计历史会话数作为序号
        existing = self._read_history()
        count = existing.count("\n## 会话 ") + 1

        header = f"\n## 会话 #{count} — {self._session_start}\n\n"
        self._append_raw(header)

        # 启动时检查是否需要浓缩
        if len(existing) > self.MAX_BUFFER_CHARS:
            self._condense_buffer()

    def _read_history(self) -> str:
        if not os.path.exists(self._history_path):
            return ""
        with open(self._history_path, "r", encoding="utf-8") as f:
            return f.read()

    def _append_raw(self, text: str):
        os.makedirs(os.path.dirname(self._history_path), exist_ok=True)
        with open(self._history_path, "a", encoding="utf-8") as f:
            f.write(text)

    def _buffer_size(self) -> int:
        return len(self._read_history())

    # ==================== 保存对话（被动，无 LLM） ====================

    def save_turn(self, session_id: str, user_input: str, agent_response: str):
        """被动保存一轮原始对话到 HISTORY.md。"""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}]\nUser: {user_input}\nAgent: {agent_response}\n\n"
        self._append_raw(entry)

        # 检查是否需要浓缩
        if self._buffer_size() > self.MAX_BUFFER_CHARS:
            threading.Thread(target=self._safe_condense, daemon=True).start()

    def _safe_condense(self):
        """加锁防并发，在后台线程执行浓缩。"""
        if self._condense_lock.acquire(blocking=False):
            try:
                self._condense_buffer()
            finally:
                self._condense_lock.release()

    # ==================== 批量浓缩 ====================

    BATCH_CONDENSE_PROMPT = (
        "你是一个对话历史归档器。以下是跨多个会话的完整对话记录。\n"
        "请完成以下任务：\n\n"
        "1. 按话题分类，将相关的多轮对话归为一组\n"
        "2. 忽略寒暄、闲聊和无信息量的对话\n"
        "3. 每组用 1-3 句中文浓缩核心信息（保留文件名、函数名、数据、决策理由）\n"
        "4. 为每组分配 2-5 字的话题标签\n\n"
        "返回 JSON 数组，每项包含 topic 和 summary：\n"
        '[{"topic": "标签", "summary": "摘要"}, ...]\n'
        "summary 中严禁出现英文双引号\"，必须用中文引号「」替代。\n"
        "如果所有对话都不值得保存，返回空数组 []。\n\n"
        "=== 对话记录 ===\n"
    )

    def _condense_buffer(self):
        """将 HISTORY.md 中的旧会话批量浓缩存入 SQLite。"""
        content = self._read_history()
        if len(content) < 500:
            return

        print("  [历史] 正在批量浓缩...")

        prompt = self.BATCH_CONDENSE_PROMPT + content[-12000:]  # 取最近的 12K 字符
        try:
            response = self._client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                stream=False,
            )
            text = response.choices[0].message.content.strip()
            entries = self._parse_batch_result(text)
        except Exception as e:
            print(f"  [历史] 批量浓缩失败: {e}")
            return

        if entries:
            self._insert_condensed(entries)
            print(f"  [历史] 浓缩完成：{len(entries)} 条记录存入 SQLite")

            # 裁剪 HISTORY.md，只保留最近的原始对话
            if len(content) > self.KEEP_RECENT_CHARS:
                trimmed = content[-(self.KEEP_RECENT_CHARS):]
                # 确保从会话边界开始裁剪
                boundary = trimmed.find("\n## 会话 ")
                if boundary > 0:
                    trimmed = trimmed[boundary:]
                with open(self._history_path, "w", encoding="utf-8") as f:
                    f.write(trimmed)

    def _parse_batch_result(self, text: str) -> list[dict]:
        """解析批量浓缩的 LLM 返回。"""
        # 去 markdown 代码块
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [e for e in data if isinstance(e, dict) and e.get("summary")]
        except json.JSONDecodeError:
            pass

        # fallback: 用正则提取 topic/summary 对
        return self._parse_batch_fallback(text)

    def _parse_batch_fallback(self, text: str) -> list[dict]:
        """正则兜底提取 topic 和 summary。"""
        entries = []
        for block in re.findall(r'\{[^}]*\}', text):
            topic_m = re.search(r'"topic"\s*:\s*"([^"]*)"', block)
            summary_start = re.search(r'"summary"\s*:\s*"', block)
            if summary_start:
                rest = block[summary_start.end():]
                last_close = rest.rstrip().rfind('"}')
                summary = rest[:last_close] if last_close >= 0 else rest.strip().rstrip('"').rstrip('}')
                if summary.strip():
                    entries.append({
                        "topic": topic_m.group(1) if topic_m else "",
                        "summary": summary.strip(),
                    })
        return entries

    # ==================== 搜索 ====================

    def search(self, query: str, limit: int = 5) -> str:
        """关键词搜索 SQLite 浓缩历史。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT session_id, timestamp, topic, summary FROM conversations "
                "WHERE summary LIKE ? OR topic LIKE ? "
                "ORDER BY timestamp DESC LIMIT ?",
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

    def get_recent(self, limit: int = 10) -> str:
        """获取最近记录：先查 SQLite，再补 HISTORY.md 最新原始对话。"""
        parts = []

        # SQLite 浓缩记录
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT topic, summary FROM conversations "
                "ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        if rows:
            parts.append("=== 浓缩历史 ===")
            for topic, summary in reversed(rows):
                topic_str = f" [{topic}]" if topic else ""
                parts.append(f"{topic_str} {summary}")

        # HISTORY.md 最近原始对话（最后几行）
        raw = self._read_history()
        if raw:
            sessions = raw.split("\n## 会话 ")
            recent_sessions = sessions[-2:]  # 最近两个会话
            recent_text = "\n## 会话 ".join(recent_sessions)
            lines = recent_text.strip().split("\n")
            tail = "\n".join(lines[-30:])  # 取最后 30 行
            if tail.strip():
                parts.append("\n=== 最近原始对话 ===")
                parts.append(tail)

        return "\n".join(parts) if parts else "暂无历史记录。"

    def get_recent_history(self, limit: int = 10) -> str:
        """获取最近历史记录（兼容旧接口）。"""
        return self.get_recent(limit=limit)
