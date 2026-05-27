import os

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL_NAME", "deepseek-v4-flash")
base_url = os.getenv("OPENAI_BASE_URL")

# 内置模型注册表
MODEL_REGISTRY = {
    "deepseek-v4-flash": {
        "name": "deepseek-v4-flash",
        "description": "轻量快速模型，适合阅读文档、简单任务、格式转换、摘要等",
        "tier": "simple",
    },
    "deepseek-v4-pro": {
        "name": "deepseek-v4-pro",
        "description": "强力推理模型，适合复杂多步骤推理、代码生成、数学计算",
        "tier": "powerful",
    },
}