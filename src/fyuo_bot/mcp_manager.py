"""
MCPManager —— 从 .fyuobot/.setting 读取已注册的 MCP 服务，统一管理多连接。

- 每个 MCP 服务在独立后台线程中维护 asyncio event loop
- 工具名自动加服务名前缀，避免冲突（如 codegraph_search, simple_mcp_mcd_test）
- 单服务连接失败不影响其他服务和 agent 正常启动
"""

import asyncio
import json
import os
import threading

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession

from tools.base import BaseTool


class MCPTool(BaseTool):
    """将 MCP 工具包装为 BaseTool。"""

    def __init__(self, name: str, description: str, parameters: dict,
                 client: "MCPServerClient"):
        self.name = name
        self.description = description
        self.parameters = parameters
        self._client = client

    def execute(self, **kwargs) -> str:
        return self._client.call_tool(self.name, kwargs)


class MCPServerClient:
    """单个 MCP 服务的后台线程客户端。"""

    def __init__(self, name: str, command: str, args: list[str],
                 timeout: float = 15.0):
        self.name = name
        self._command = command
        self._args = args
        self._timeout = timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: ClientSession | None = None
        self._ready = threading.Event()
        self._error: str = ""

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        connected = self._ready.wait(timeout=timeout)
        if not connected:
            raise RuntimeError(
                f"[{name}] 连接超时 ({timeout}s)"
                f"{' — ' + self._error if self._error else ''}"
            )

    @property
    def _original_names(self) -> dict[str, str]:
        """全限定名 → 原始工具名的映射。"""
        if not hasattr(self, "_orig_map"):
            self._orig_map: dict[str, str] = {}
        return self._orig_map

    # ---- 后台循环 ----

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect())
        except Exception as e:
            self._error = str(e)
            self._ready.set()
            return
        self._loop.run_forever()
        try:
            self._loop.run_until_complete(self._cleanup())
        except Exception:
            pass

    async def _connect(self):
        params = StdioServerParameters(
            command=self._command,
            args=self._args,
        )
        self._transport_ctx = stdio_client(params)
        read, write = await self._transport_ctx.__aenter__()
        self._session_ctx = ClientSession(read, write)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()
        self._ready.set()

    async def _cleanup(self):
        if self._session_ctx is not None:
            await self._session_ctx.__aexit__(None, None, None)
            self._session_ctx = None
        if self._transport_ctx is not None:
            await self._transport_ctx.__aexit__(None, None, None)
            self._transport_ctx = None

    # ---- 同步接口 ----

    def call_tool(self, qualified_name: str, arguments: dict) -> str:
        # 还原原始工具名
        original_name = self._original_names.get(qualified_name, qualified_name)

        async def _call():
            result = await self._session.call_tool(original_name, arguments)
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            return "\n".join(parts) if parts else str(result.content)

        future = asyncio.run_coroutine_threadsafe(_call(), self._loop)
        return future.result(timeout=self._timeout)

    def list_tools(self) -> list[dict]:
        async def _list():
            result = await self._session.list_tools()
            return [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema if hasattr(t, "inputSchema") else {},
                }
                for t in result.tools
            ]

        future = asyncio.run_coroutine_threadsafe(_list(), self._loop)
        return future.result(timeout=self._timeout)

    def close(self):
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)


class MCPManager:
    """统一管理多个 MCP 服务连接，从 .setting 读取配置。"""

    SETTING_PATH = ".fyuobot/.setting"

    def __init__(self, workspace: str, timeout: float = 15.0):
        self._clients: list[MCPServerClient] = []
        self._tools: list[BaseTool] = []
        self._workspace = workspace
        self._timeout = timeout
        self._load_and_connect()

    # ---- 加载配置 ----

    def _load_config(self) -> list[dict]:
        path = os.path.join(self._workspace, self.SETTING_PATH)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("mcp_servers", [])

    # ---- 连接所有服务 ----

    def _load_and_connect(self):
        servers = self._load_config()
        for cfg in servers:
            if not cfg.get("enabled", True):
                continue
            name = cfg["name"]
            command = cfg["command"]
            args = cfg.get("args", [])
            prefix = name.replace("-", "_")

            try:
                client = MCPServerClient(name, command, args, timeout=self._timeout)
            except Exception as e:
                print(f"  [MCP] {name}: 连接失败 — {e}")
                continue

            # 发现工具并加前缀
            try:
                raw_tools = client.list_tools()
            except Exception as e:
                print(f"  [MCP] {name}: 发现工具失败 — {e}")
                continue

            for t in raw_tools:
                qualified = t["name"] if t["name"].startswith(prefix) else f"{prefix}_{t['name']}"
                client._original_names[qualified] = t["name"]
                tool = MCPTool(
                    name=qualified,
                    description=f"[{name}] {t['description']}",
                    parameters=t["parameters"],
                    client=client,
                )
                self._tools.append(tool)

            self._clients.append(client)
            print(f"  [MCP] {name}: 已连接，{len(raw_tools)} 个工具")

    # ---- 外部接口 ----

    @property
    def tools(self) -> list[BaseTool]:
        return self._tools

    def close_all(self):
        for c in self._clients:
            c.close()
