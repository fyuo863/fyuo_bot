# fyuo_bot

基于 ReAct 循环的 AI Agent 框架，支持多轮自省、工具调用、子 Agent 嵌套、记忆持久化与对话历史浓缩。

## 项目架构

```
src/fyuo_bot/
├── core/               # 核心引擎
│   ├── agent.py        # ReActAgent —— 思考-行动循环，自评机制
│   ├── chat.py         # AgentChat —— OpenAI 流式对话 + tool calling 解析
│   └── usage.py        # Usage —— 流式碎片拼装为 assistant 消息
├── tools/              # 工具系统
│   ├── base.py         # BaseTool 基类 + 内置工具（文件、命令、天气等）
│   ├── agent_tool.py   # AgentTool —— 将整个 Agent 封装为可嵌套调用的工具
│   └── memory_tools.py # replace_memory / get_history 工具
├── memory/             # 存储系统
│   ├── manager.py      # MemoryManager —— MEMORY.md / USER.md 持久记忆
│   └── history.py      # HistoryManager —— SQLite 对话历史，LLM 自动浓缩分类
├── config/
│   └── config.py       # API 密钥、模型注册表
└── main.py             # CLI 入口
```

## 核心特性

### ReAct 思考-行动循环

Agent 遵循 Reasoning + Acting 模式：思考 → 调用工具 → 观察结果 → 继续思考 → 完成任务。最大迭代 20 轮，防止死循环。

### 多轮自评机制

Agent 产出答案后，系统自动注入 `[自动自评]` 提示，让 Agent 审视自己回答的完整性和正确性。发现不足可调用工具修正，确认无误回复"确认完成"退出。最多自评 3 轮，最终返回自评前的干净答案。

### 子 Agent 嵌套

`AgentTool` 将整个 ReAct Agent 封装为工具，支持多层嵌套调用，每层递减深度上限（默认 3 层），底层 Agent 收到深度警告后必须直接完成任务。

### 双轨存储

| 存储 | 位置 | 内容 | 管理方式 |
|---|---|---|---|
| 持久记忆 | `MEMORY.md` / `USER.md` | 操作习惯、个人信息 | Agent 调用 `replace_memory` |
| 对话历史 | SQLite `history.db` | 浓缩对话 + 话题分类 | 每轮自动 `save_turn`，`get_history` 搜索 |

对话历史存入前经 LLM 判断：寒暄/闲聊自动跳过，有实质内容则按话题分类（如 "Python编程"、"Bug修复"）并浓缩为 1-3 句摘要。

### 内置工具

| 工具 | 用途 |
|---|---|
| `new_file` | 创建文件或文件夹，自检确认 |
| `write_file` | Python 原生文件流写入，写后读回校验 |
| `read_file` | 读取工作区内文件 |
| `list_files` | 列出目录结构 |
| `do_command` | 执行 shell 命令（需用户审批） |
| `get_weather` | 查询天气 |
| `get_location` | IP 定位 |
| `let_user_answer` | 向用户提问 |
| `run_agent` | 启动子 Agent |
| `get_model_list` | 查看可用模型 |
| `replace_memory` | 管理持久记忆 |
| `get_history` | 搜索对话历史 |

## 快速开始

```bash
# 设置环境变量
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
export OPENAI_MODEL_NAME="deepseek-v4-pro"

# 运行
python src/fyuo_bot/main.py
```

## 当前阶段

核心功能已可用：ReAct 循环、工具调用、多轮自评、子 Agent 嵌套、持久记忆、对话历史浓缩。后续方向：

- [ ] 异步工具执行
- [ ] 对话历史向量搜索
- [ ] 更完善的安全沙箱
- [ ] 外部工具支持
