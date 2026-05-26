import os
import chromadb
from chromadb.utils import embedding_functions

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


class MemoryEngine:
    def __init__(self, db_path: str = "", host: str = "", port: int = 0):
        # 绕过 Windows 系统代理，否则 httpx 走代理会导致 502
        if "NO_PROXY" not in os.environ:
            os.environ["NO_PROXY"] = "127.0.0.1,localhost"
        # 连接模式：优先 HTTP 远程（Docker），否则本地持久化
        if host or port:
            self.client = chromadb.HttpClient(
                host=host or DEFAULT_HOST,
                port=port or DEFAULT_PORT,
            )
            print(f" 已连接远程 ChromaDB: {host or DEFAULT_HOST}:{port or DEFAULT_PORT}")
        elif db_path:
            self.client = chromadb.PersistentClient(path=db_path)
            print(f" 使用本地 ChromaDB: {db_path}")
        else:
            self.client = chromadb.PersistentClient(path="./AgentMemory")
            print(" 使用默认本地 ChromaDB: ./AgentMemory")

        self.embedding_func = embedding_functions.OllamaEmbeddingFunction(
            url="http://127.0.0.1:11435",
            model_name="nomic-embed-text",
        )
        print(" embedding 函数已连接 Ollama (nomic-embed-text)")

        self.collection = self.client.get_or_create_collection(
            name="agent_history",
            embedding_function=self.embedding_func,
        )

    def save_memory(self, session_id: str, user_input: str, agent_response: str):
        memory_text = f"任务: {user_input}\n总结: {agent_response}"
        self.collection.add(
            documents=[memory_text],
            metadatas=[{"type": "task_summary", "user_query": user_input}],
            ids=[session_id],
        )
        print("[记忆引擎] 本次任务总结已成功存入本地大脑。")

    def retrieve_memory(self, current_input: str, top_k: int = 5) -> str:
        if self.collection.count() == 0:
            return ""
        results = self.collection.query(
            query_texts=[current_input],
            n_results=top_k,
        )
        memories = results["documents"][0]
        if not memories:
            return ""
        background = "\n".join([f"- {m}" for m in memories])
        return f"\n\n[相关历史记忆（补充背景）]\n{background}\n"
