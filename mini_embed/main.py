from fastapi import FastAPI
from pydantic import BaseModel
from fastembed import TextEmbedding
import uvicorn

app = FastAPI()

# 启动时自动下载并加载极轻量级中文模型 bge-small-zh-v1.5 (仅 ~130MB)
print("正在加载极简版 Embedding 模型...")
embedding_model = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5")
print("模型加载完毕，服务启动！")

class EmbedRequest(BaseModel):
    model: str
    prompt: str

@app.post("/api/embeddings")
async def generate_embedding(req: EmbedRequest):
    # FastEmbed 接收列表，返回生成器，我们取第一个结果
    vectors = list(embedding_model.embed([req.prompt]))
    return {
        # 将 numpy 数组转为普通 Python 列表返回
        "embedding": vectors[0].tolist()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11434)