FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .

# 使用清华源加速安装
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY main.py .

# 暴露与您之前 Ollama 相同的端口
EXPOSE 11435

CMD ["python", "main.py"]