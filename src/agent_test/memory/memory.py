import chromadb
from chromadb.utils import embedding_functions
import os

class MemoryEngine:
    def __init__(self, db_path="./AgentMemory"):
        # 1. 初始化本地持久化数据库
        self.client = chromadb.PersistentClient(path=db_path)
        
        # 2. 加载纯本地的中文友好型 Embedding 模型
        # 'paraphrase-multilingual-MiniLM-L12-v2' 体积小且对中文支持极好
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # 3. 获取或创建名为 "agent_history" 的记忆集合
        self.collection = self.client.get_or_create_collection(
            name="agent_history",
            embedding_function=self.embedding_func
        )

    def save_memory(self, session_id: str, user_input: str, agent_response: str):
        """将一轮完整的对话存入长期记忆"""
        # 将用户的提问和 Agent 的执行结果拼接成一段完整的记忆
        memory_text = f"User asked: {user_input}\nAgent resolved: {agent_response}"
        
        self.collection.add(
            documents=[memory_text],
            metadatas=[{"type": "conversation", "user_query": user_input}],
            ids=[session_id] # 使用时间戳或唯一ID
        )
        print("💾 [记忆引擎] 本次任务经验已成功存入本地大脑。")

    def retrieve_memory(self, current_input: str, top_k: int = 2) -> str:
        """根据当前问题，唤醒最相关的历史记忆"""
        if self.collection.count() == 0:
            return "" # 还没产生过记忆

        results = self.collection.query(
            query_texts=[current_input],
            n_results=top_k
        )
        
        memories = results['documents'][0]
        if not memories:
            return ""
            
        # 将查找到的历史记忆拼装成背景字符串
        background = "\n".join([f"- {m}" for m in memories])
        return f"\n\n[相关历史记忆（补充背景）]\n{background}\n"