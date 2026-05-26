from fastapi import FastAPI
from pydantic import BaseModel
from fastembed import TextEmbedding
from typing import Union, List, Optional
import uvicorn

app = FastAPI()

print("正在加载极简版 Embedding 模型...")
# 这里会自动下载并使用 bge-small-zh-v1.5
embedding_model = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5")
print("模型加载完毕，服务启动！")

# 兼容新旧版本 Ollama 的数据结构
class EmbedRequest(BaseModel):
    model: str
    prompt: Optional[Union[str, List[str]]] = None  # 旧版使用 prompt
    input: Optional[Union[str, List[str]]] = None   # 新版使用 input

# 同时注册两个路由，彻底兼容所有版本的 Ollama 客户端
@app.post("/api/embed")
@app.post("/api/embeddings")
async def generate_embedding(req: EmbedRequest):
    # 提取文本：优先取 input，如果没有再取 prompt
    texts = req.input or req.prompt
    
    # FastEmbed 需要列表格式，如果是单条字符串则转为列表
    if isinstance(texts, str):
        texts = [texts]
        
    # 生成向量
    vectors = list(embedding_model.embed(texts))
    
    # 同时返回新老格式，满足各种客户端的解析需求
    return {
        "embedding": vectors[0].tolist(),               # 满足旧版解析
        "embeddings": [v.tolist() for v in vectors]     # 满足新版解析
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11435)