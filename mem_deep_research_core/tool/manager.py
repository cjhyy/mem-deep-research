import asyncio
import functools
import os
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeVar

from mcp import ClientSession, StdioServerParameters  # (already imported in config.py)
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

# 尝试导入 streamable-http 客户端（MCP SDK >= 1.1.0）
try:
    from mcp.client.streamable_http import streamablehttp_client

    HAS_STREAMABLE_HTTP = True
except ImportError:
    HAS_STREAMABLE_HTTP = False
    streamablehttp_client = None

from mem_deep_research_core.core.secure_context import get_real_value
from mem_deep_research_core.exceptions import (
    ToolError,
)
from mem_deep_research_core.mem_deep_research_logging.logger import bootstrap_logger

from .mcp_servers.browser_session import PlaywrightSession

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
logger = bootstrap_logger(level=LOGGER_LEVEL)

R = TypeVar("R")


def _default_env_inject(ctx) -> Any:
    """默认的环境变量注入逻辑"""
    from mem_deep_research_core.mem_deep_research_logging.logger import TASK_CONTEXT_VAR

    server_params = ctx.server_params
    context = ctx.context

    # Inject task ID
    if TASK_CONTEXT_VAR.get() is not None:
        server_params.env["TASK_ID"] = TASK_CONTEXT_VAR.get()

    # Inject user context if provided（优先从 _secure 取真实值）
    if context:
        for key in ["user_id", "org_id", "room_id", "timezone", "trace_id"]:
            val = get_real_value(context, key)
            if val:
                server_params.env[key.upper()] = val

    return server_params


def update_server_params_with_context(
    server_params: StdioServerParameters,
    context: dict[str, Any] | None = None,
) -> StdioServerParameters:
    """
    Update the server params with the task context and user context.

    Injects environment variables into the MCP subprocess.
    Supports hooks for custom env injection.

    Args:
        server_params: The server parameters to update
        context: User context dict with user_id, org_id, room_id, timezone, trace_id
    """
    from mem_deep_research_core.core.hooks import HookContext, hooks

    # 设置默认实现
    hooks.set_default("on_env_inject", _default_env_inject)

    # 创建上下文并调用钩子
    ctx = HookContext(
        hook_name="on_env_inject",
        server_params=server_params,
        context=context,
    )

    return hooks.call("on_env_inject", ctx)


def with_timeout(timeout_s: float = 300.0):
    """
    Decorator: wraps any *async* function in asyncio.wait_for().
    Usage:
        @with_timeout(20)
        async def create_message_foo(...): ...
    """

    def decorator(
        func: Callable[..., Awaitable[R]],
    ) -> Callable[..., Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> R:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_s)

        return wrapper

    return decorator


class ToolManagerProtocol(Protocol):
    """this enables other kinds of tool manager."""

    async def get_all_tool_definitions(self) -> Any: ...
    async def execute_tool_call(
        self, *, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> Any: ...


class ToolManager(ToolManagerProtocol):
    # Class-level cache for tool definitions (shared across instances)
    _tool_definitions_cache: dict[
        str, tuple[float, list]
    ] = {}  # {server_name: (timestamp, definitions)}
    _CACHE_TTL_SECONDS: float = 300.0  # 5 minutes cache TTL

    def __init__(self, server_configs, tool_blacklist=None, cache_ttl: float = 300.0):
        """
        Initialize ToolManager.
        :param server_configs: List returned by create_server_parameters()
            每个 config 可以是:
            - {"name": "xxx", "params": StdioServerParameters(...)}  # 本地 stdio
            - {"name": "xxx", "params": "http://...", "transport": "sse"}  # 远程 SSE
            - {"name": "xxx", "params": "http://...", "transport": "streamable-http"}  # 远程 HTTP
        :param tool_blacklist: Set of (server_name, tool_name) tuples to blacklist
        :param cache_ttl: Tool definitions cache TTL in seconds (default: 300)
        """
        self.server_configs = server_configs
        # 存储完整配置，包括 transport 和 headers
        self.server_dict = {config["name"]: config["params"] for config in server_configs}
        # 存储传输方式、headers 和 inject_context 配置
        self.server_transport = {
            config["name"]: config.get("transport", "stdio") for config in server_configs
        }
        self.server_headers = {
            config["name"]: config.get("headers", {}) for config in server_configs
        }
        self.server_inject_context = {
            config["name"]: config.get("inject_context", True) for config in server_configs
        }
        self.browser_session = None
        self._context: dict[str, Any] = {}  # User context for MCP tool calls
        self.tool_blacklist = tool_blacklist if tool_blacklist else set()
        self._cache_ttl = cache_ttl

        logger.info(f"ToolManager initialized, loaded servers: {list(self.server_dict.keys())}")

    def _is_cache_valid(self, server_name: str) -> bool:
        """Check if the cached tool definitions for a server are still valid."""
        import time

        if server_name not in self._tool_definitions_cache:
            return False
        timestamp, _ = self._tool_definitions_cache[server_name]
        return (time.time() - timestamp) < self._cache_ttl

    def _get_cached_definitions(self, server_name: str) -> list | None:
        """Get cached tool definitions for a server if valid."""
        if self._is_cache_valid(server_name):
            _, definitions = self._tool_definitions_cache[server_name]
            logger.debug(f"[ToolManager] Using cached definitions for '{server_name}'")
            return definitions
        return None

    def _cache_definitions(self, server_name: str, definitions: list) -> None:
        """Cache tool definitions for a server."""
        import time

        self._tool_definitions_cache[server_name] = (time.time(), definitions)
        logger.debug(f"[ToolManager] Cached {len(definitions)} definitions for '{server_name}'")

    def clear_cache(self, server_name: str | None = None) -> None:
        """Clear tool definitions cache for a specific server or all servers."""
        if server_name:
            self._tool_definitions_cache.pop(server_name, None)
            logger.info(f"[ToolManager] Cleared cache for '{server_name}'")
        else:
            self._tool_definitions_cache.clear()
            logger.info("[ToolManager] Cleared all tool definitions cache")

    def set_context(self, context: dict[str, Any] | None) -> None:
        """
        Set the user context for MCP tool calls.

        Args:
            context: Dict containing user_id, org_id, room_id, timezone, trace_id
        """
        self._context = context or {}

    def _extract_tool_result(self, tool_name: str, tool_result, arguments: dict) -> str:
        """
        从工具结果中提取文本内容。

        Args:
            tool_name: 工具名称
            tool_result: MCP 工具返回的结果
            arguments: 工具调用参数

        Returns:
            提取的文本内容
        """
        result_content = ""

        if tool_result.content and len(tool_result.content) > 0:
            text_content = tool_result.content[-1].text
            if text_content is not None and text_content.strip():
                result_content = text_content
            else:
                result_content = f"Tool '{tool_name}' completed but returned empty text - this may be expected or indicate an issue"
        else:
            result_content = f"Tool '{tool_name}' completed but returned no content - this may be expected or indicate an issue"

        # If result is empty, log warning
        if not tool_result.content:
            logger.error(
                f"Tool '{tool_name}' returned empty content, tool_result.content: {tool_result.content}"
            )

        # post hoc check for browsing agent reading answers from hf datasets
        if self._should_block_hf_scraping(tool_name, arguments):
            result_content = "You are trying to scrape a Hugging Face dataset for answers, please do not use the scrape tool for this purpose."

        return result_content

    def _is_huggingface_dataset_or_space_url(self, url):
        """
        Check if the URL is a Hugging Face dataset or space URL.
        :param url: The URL to check
        :return: True if it's a HuggingFace dataset or space URL, False otherwise
        """
        if not url:
            return False
        return "huggingface.co/datasets" in url or "huggingface.co/spaces" in url

    def _should_block_hf_scraping(self, tool_name, arguments):
        """
        Check if we should block scraping of Hugging Face datasets/spaces.
        :param tool_name: The name of the tool being called
        :param arguments: The arguments passed to the tool
        :return: True if scraping should be blocked, False otherwise
        """
        return (
            tool_name == "scrape"
            and arguments.get("url")
            and self._is_huggingface_dataset_or_space_url(arguments["url"])
        )

    def get_server_params(self, server_name):
        """Get parameters for specified server"""
        return self.server_dict.get(server_name)

    async def _find_servers_with_tool(self, tool_name):
        """
        Find servers containing the specified tool name among all servers
        :param tool_name: Tool name to search for
        :return: List of server names containing the tool
        """
        servers_with_tool = []

        for config in self.server_configs:
            server_name = config["name"]
            server_params = config["params"]

            try:
                if isinstance(server_params, StdioServerParameters):
                    async with (
                        stdio_client(
                            update_server_params_with_context(server_params, self._context)
                        ) as (read, write),
                        ClientSession(read, write, sampling_callback=None) as session,
                    ):
                        await session.initialize()
                        tools_response = await session.list_tools()
                        # Follow the same blacklist logic as get_all_tool_definitions
                        for tool in tools_response.tools:
                            if (server_name, tool.name) in self.tool_blacklist:
                                continue
                            if tool.name == tool_name:
                                servers_with_tool.append(server_name)
                                break
                elif isinstance(server_params, str) and server_params.startswith(
                    ("http://", "https://")
                ):
                    # SSE endpoint
                    async with sse_client(server_params) as (read, write):
                        async with ClientSession(read, write, sampling_callback=None) as session:
                            await session.initialize()
                            tools_response = await session.list_tools()
                            for tool in tools_response.tools:
                                # Consistent with get_all_tool_definitions: SSE part has no blacklist processing
                                # Can add specific tool filtering logic here (if needed)
                                # if server_name == "tool-excel" and tool.name not in ["get_workbook_metadata", "read_data_from_excel"]:
                                #     continue
                                if tool.name == tool_name:
                                    servers_with_tool.append(server_name)
                                    break
                else:
                    logger.error(
                        f"Error: Unknown parameter type for server '{server_name}': {type(server_params)}"
                    )
                    # For unknown types, we skip rather than throw an exception, because this is a search function
                    continue
            except Exception as e:
                logger.error(
                    f"Error: Cannot connect or get tools from server '{server_name}' to find '{tool_name}': {e}"
                )
                continue

        return servers_with_tool

    async def _fetch_single_server_definitions(self, config: dict) -> dict:
        """
        Fetch tool definitions from a single server.
        Returns dict with 'name' and 'tools' keys.

        支持三种传输模式:
        - stdio: 本地子进程
        - sse: 远程 SSE 端点
        - streamable-http: 远程 HTTP 端点 (推荐)
        """
        server_name = config["name"]
        server_params = config["params"]
        transport = config.get("transport", "stdio")
        headers = config.get("headers", {})
        one_server_for_prompt = {"name": server_name, "tools": []}

        # Check cache first
        cached = self._get_cached_definitions(server_name)
        if cached is not None:
            one_server_for_prompt["tools"] = cached
            return one_server_for_prompt

        logger.info(
            f"Getting tool definitions for server '{server_name}' (transport={transport})..."
        )

        try:
            if isinstance(server_params, StdioServerParameters):
                # 本地 stdio 模式
                async with stdio_client(
                    update_server_params_with_context(server_params, self._context)
                ) as (read, write):
                    async with ClientSession(read, write, sampling_callback=None) as session:
                        await session.initialize()
                        tools_response = await session.list_tools()
                        # black list some tools
                        for tool in tools_response.tools:
                            if (server_name, tool.name) in self.tool_blacklist:
                                logger.info(
                                    f"Tool '{tool.name}' in server '{server_name}' is blacklisted, skipping."
                                )
                                continue
                            one_server_for_prompt["tools"].append(
                                {
                                    "name": tool.name,
                                    "description": tool.description,
                                    "schema": tool.inputSchema,
                                }
                            )
            elif isinstance(server_params, str) and server_params.startswith(
                ("http://", "https://")
            ):
                # 远程模式
                if transport == "streamable-http" and HAS_STREAMABLE_HTTP:
                    # Streamable HTTP 模式 (推荐)
                    async with streamablehttp_client(server_params, headers=headers) as (
                        read,
                        write,
                        _,
                    ):
                        async with ClientSession(read, write, sampling_callback=None) as session:
                            await session.initialize()
                            tools_response = await session.list_tools()
                            for tool in tools_response.tools:
                                if (server_name, tool.name) in self.tool_blacklist:
                                    continue
                                one_server_for_prompt["tools"].append(
                                    {
                                        "name": tool.name,
                                        "description": tool.description,
                                        "schema": tool.inputSchema,
                                    }
                                )
                else:
                    # SSE 模式 (回退)
                    async with sse_client(server_params) as (read, write):
                        async with ClientSession(read, write, sampling_callback=None) as session:
                            await session.initialize()
                            tools_response = await session.list_tools()
                            for tool in tools_response.tools:
                                if (server_name, tool.name) in self.tool_blacklist:
                                    continue
                                one_server_for_prompt["tools"].append(
                                    {
                                        "name": tool.name,
                                        "description": tool.description,
                                        "schema": tool.inputSchema,
                                    }
                                )
            else:
                logger.error(
                    f"Error: Unknown parameter type for server '{server_name}': {type(server_params)}"
                )
                raise TypeError(
                    f"Unknown server params type for {server_name}: {type(server_params)}"
                )

            logger.info(
                f"Successfully obtained {len(one_server_for_prompt['tools'])} tool definitions for server '{server_name}'."
            )
            # Cache the definitions
            self._cache_definitions(server_name, one_server_for_prompt["tools"])

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Connection error for server '{server_name}': {e}")
            one_server_for_prompt["tools"] = [{"error": f"Connection failed: {e}"}]
        except (TypeError, ValueError, RuntimeError) as e:
            import traceback

            logger.error(
                f"Error: Cannot connect or get tools from server '{server_name}': {e}\n"
                f"Server params type: {type(server_params)}\n"
                f"Server params: {server_params if isinstance(server_params, str) else 'StdioServerParameters'}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            one_server_for_prompt["tools"] = [{"error": f"Failed to fetch tools: {e}"}]

        return one_server_for_prompt

    async def get_all_tool_definitions(self, parallel: bool = True):
        """
        Connect to all configured servers and get their tool definitions.
        Returns a list suitable for passing to Prompt generators.

        Args:
            parallel: If True, fetch from all servers concurrently. Default: True.
        """
        if not self.server_configs:
            return []

        if parallel and len(self.server_configs) > 1:
            # Parallel fetching for better performance
            tasks = [
                self._fetch_single_server_definitions(config) for config in self.server_configs
            ]
            all_servers_for_prompt = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions from gather
            result = []
            for i, server_result in enumerate(all_servers_for_prompt):
                if isinstance(server_result, Exception):
                    server_name = self.server_configs[i]["name"]
                    logger.error(f"Parallel fetch failed for '{server_name}': {server_result}")
                    result.append({"name": server_name, "tools": [{"error": str(server_result)}]})
                else:
                    result.append(server_result)
            return result
        else:
            # Sequential fetching (original behavior)
            all_servers_for_prompt = []
            for config in self.server_configs:
                server_result = await self._fetch_single_server_definitions(config)
                all_servers_for_prompt.append(server_result)
            return all_servers_for_prompt

    @with_timeout(900)
    async def execute_tool_call(
        self, server_name, tool_name, arguments, context: dict = None
    ) -> Any:
        """
        Execute a single tool call.
        :param server_name: Server name
        :param tool_name: Tool name
        :param arguments: Tool arguments dictionary
        :param context: Optional request context containing user_id, org_id, etc.
                       This is the MOST RELIABLE way to pass context in concurrent scenarios.
        :return: Dictionary containing result or error
        """
        server_params = self.get_server_params(server_name)
        if not server_params:
            logger.error(f"Error: Attempting to call server '{server_name}' that was not found")

            # Try to find the tool in all available servers
            suggested_servers = await self._find_servers_with_tool(tool_name)

            error_message = f"Server '{server_name}' not found."

            if len(suggested_servers) == 1:
                # Auto-correction: only one server contains the tool, try to auto-correct and execute
                correct_server = suggested_servers[0]
                logger.info(
                    f"Auto-correction: Server '{server_name}' not found, but found tool '{tool_name}' in '{correct_server}', trying to auto-correct and execute"
                )

                try:
                    # Recursive call, using the correct server name
                    corrected_result = await self.execute_tool_call(
                        correct_server, tool_name, arguments
                    )

                    # If auto-correction is successful, add a note in the result
                    if "result" in corrected_result:
                        # Add auto-correction note in the result, including the reason for the correction
                        correction_note = f"[Auto-corrected: Server '{server_name}' not found, but tool '{tool_name}' was found only in server '{correct_server}', so automatically used '{correct_server}' instead] "
                        corrected_result["result"] = correction_note + str(
                            corrected_result["result"]
                        )
                        return corrected_result
                    elif "error" in corrected_result:
                        # If there is an error after auto-correction, add a note in the error message
                        correction_note = f"[Auto-corrected: Server '{server_name}' not found, but tool '{tool_name}' was found only in server '{correct_server}', attempted auto-correction but still failed] "
                        corrected_result["error"] = correction_note + str(corrected_result["error"])
                        return corrected_result

                except (ConnectionError, TimeoutError, OSError) as auto_correct_error:
                    logger.error(
                        f"Auto-correction failed due to connection error: {auto_correct_error}"
                    )
                    error_message += f" Found tool '{tool_name}' in server '{correct_server}' and attempted auto-correction, but connection failed: {str(auto_correct_error)}"
                except (ToolError, RuntimeError) as auto_correct_error:
                    logger.error(f"Auto-correction failed: {auto_correct_error}")
                    error_message += f" Found tool '{tool_name}' in server '{correct_server}' and attempted auto-correction, but it failed: {str(auto_correct_error)}"

            elif len(suggested_servers) > 1:
                error_message += f" However, found tool '{tool_name}' in these servers: {', '.join(suggested_servers)}. You may want to use one of these servers instead."
            else:
                error_message += (
                    " It is possible that the server_name and tool_name were confused or mixed up. "
                    "You should try again and carefully check the server name and tool name provided in the system prompt."
                )

            return {
                "server_name": server_name,
                "tool_name": tool_name,
                "error": error_message,
            }

        logger.info(
            f"Connecting to server '{server_name}' to call tool '{tool_name}'...call arguments: '{arguments}'..."
        )

        if server_name == "playwright":
            try:
                if self.browser_session is None:
                    self.browser_session = PlaywrightSession(server_params)
                    await self.browser_session.connect()
                tool_result = await self.browser_session.call_tool(tool_name, arguments=arguments)

                # Check if result is empty and provide better feedback
                if tool_result is None or tool_result == "":
                    logger.error(
                        f"Tool '{tool_name}' returned empty result, this may be normal (such as delete operations) or the tool execution may have issues"
                    )
                    return {
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "result": f"Tool '{tool_name}' returned empty result - this may be expected (e.g., delete operations) or indicate an issue with tool execution",
                    }

                return {
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "result": tool_result,
                }
            except Exception as e:
                return {
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "error": f"Tool call failed: {str(e)}",
                }
        else:
            try:
                result_content = None

                # CRITICAL: Use explicitly passed context as HIGHEST PRIORITY
                # This is the only reliable way in concurrent scenarios because:
                # - context param: passed directly from Orchestrator.self.context (request-specific, MOST RELIABLE)
                effective_context = context or self._context or {}

                enriched_arguments = {**arguments}

                # _mcp_context 注入（用于支持该协议的 MCP 服务器）
                # inject_context=false 的服务器跳过（如使用严格 Pydantic 验证的服务器）
                # 优先从 _secure 取真实值
                should_inject = self.server_inject_context.get(server_name, True)
                user_id_to_inject = get_real_value(effective_context, "user_id")
                if user_id_to_inject and should_inject:
                    enriched_arguments["_mcp_context"] = {
                        "user_id": user_id_to_inject,
                        "org_id": get_real_value(effective_context, "org_id"),
                        "room_id": get_real_value(effective_context, "room_id"),
                        "timezone": get_real_value(effective_context, "timezone") or "UTC",
                        "trace_id": get_real_value(effective_context, "trace_id"),
                    }

                if isinstance(server_params, StdioServerParameters):
                    # Note: User context is now passed via arguments["_mcp_context"] (primary)
                    # Environment variables are still set for backwards compatibility
                    # The MCP server (discovery_tools.py) reads _mcp_context first, then fallback to env vars

                    async with stdio_client(
                        update_server_params_with_context(server_params, effective_context)
                    ) as (read, write):
                        async with ClientSession(read, write, sampling_callback=None) as session:
                            await session.initialize()
                            try:
                                tool_result = await session.call_tool(
                                    tool_name, arguments=enriched_arguments
                                )
                                # Safely extract result content without changing original format
                                if tool_result.content and len(tool_result.content) > 0:
                                    text_content = tool_result.content[-1].text
                                    if text_content is not None and text_content.strip():
                                        result_content = text_content  # Preserve original format!
                                    else:
                                        result_content = f"Tool '{tool_name}' completed but returned empty text - this may be expected or indicate an issue"
                                else:
                                    result_content = f"Tool '{tool_name}' completed but returned no content - this may be expected or indicate an issue"

                                # If result is empty, log warning
                                if not tool_result.content:
                                    logger.error(
                                        f"Tool '{tool_name}' returned empty content, tool_result.content: {tool_result.content}"
                                    )

                                # post hoc check for browsing agent reading answers from hf datsets
                                if self._should_block_hf_scraping(tool_name, arguments):
                                    result_content = "You are trying to scrape a Hugging Face dataset for answers, please do not use the scrape tool for this purpose."
                            except TimeoutError as tool_error:
                                logger.error(f"Tool execution timeout: {tool_error}")
                                return {
                                    "server_name": server_name,
                                    "tool_name": tool_name,
                                    "error": f"Tool execution timed out: {str(tool_error)}",
                                }
                            except (ConnectionError, OSError) as tool_error:
                                logger.error(f"Tool connection error: {tool_error}")
                                return {
                                    "server_name": server_name,
                                    "tool_name": tool_name,
                                    "error": f"Tool connection failed: {str(tool_error)}",
                                }
                            except (ValueError, TypeError, RuntimeError) as tool_error:
                                logger.error(f"Tool execution error: {tool_error}")
                                return {
                                    "server_name": server_name,
                                    "tool_name": tool_name,
                                    "error": f"Tool execution failed: {str(tool_error)}",
                                }
                elif isinstance(server_params, str) and server_params.startswith(
                    ("http://", "https://")
                ):
                    # 获取传输模式和 headers
                    transport = self.server_transport.get(server_name, "sse")
                    headers = self.server_headers.get(server_name, {})

                    # 根据传输模式选择客户端
                    if transport == "streamable-http" and HAS_STREAMABLE_HTTP:
                        # Streamable HTTP 模式
                        async with streamablehttp_client(server_params, headers=headers) as (
                            read,
                            write,
                            _,
                        ):
                            async with ClientSession(
                                read, write, sampling_callback=None
                            ) as session:
                                await session.initialize()
                                try:
                                    tool_result = await session.call_tool(
                                        tool_name, arguments=enriched_arguments
                                    )
                                    result_content = self._extract_tool_result(
                                        tool_name, tool_result, arguments
                                    )
                                except TimeoutError as tool_error:
                                    logger.error(f"Tool execution timeout (HTTP): {tool_error}")
                                    return {
                                        "server_name": server_name,
                                        "tool_name": tool_name,
                                        "error": f"Tool execution timed out: {str(tool_error)}",
                                    }
                                except (ConnectionError, OSError) as tool_error:
                                    logger.error(f"Tool connection error (HTTP): {tool_error}")
                                    return {
                                        "server_name": server_name,
                                        "tool_name": tool_name,
                                        "error": f"Tool connection failed: {str(tool_error)}",
                                    }
                                except (ValueError, TypeError, RuntimeError) as tool_error:
                                    logger.error(f"Tool execution error (HTTP): {tool_error}")
                                    return {
                                        "server_name": server_name,
                                        "tool_name": tool_name,
                                        "error": f"Tool execution failed: {str(tool_error)}",
                                    }
                    else:
                        # SSE 模式（回退）
                        async with sse_client(server_params) as (read, write):
                            async with ClientSession(
                                read, write, sampling_callback=None
                            ) as session:
                                await session.initialize()
                                try:
                                    tool_result = await session.call_tool(
                                        tool_name, arguments=enriched_arguments
                                    )
                                    result_content = self._extract_tool_result(
                                        tool_name, tool_result, arguments
                                    )
                                except TimeoutError as tool_error:
                                    logger.error(f"Tool execution timeout (SSE): {tool_error}")
                                    return {
                                        "server_name": server_name,
                                        "tool_name": tool_name,
                                        "error": f"Tool execution timed out: {str(tool_error)}",
                                    }
                                except (ConnectionError, OSError) as tool_error:
                                    logger.error(f"Tool connection error (SSE): {tool_error}")
                                    return {
                                        "server_name": server_name,
                                        "tool_name": tool_name,
                                        "error": f"Tool connection failed: {str(tool_error)}",
                                    }
                                except (ValueError, TypeError, RuntimeError) as tool_error:
                                    logger.error(f"Tool execution error (SSE): {tool_error}")
                                    return {
                                        "server_name": server_name,
                                        "tool_name": tool_name,
                                        "error": f"Tool execution failed: {str(tool_error)}",
                                    }
                else:
                    raise TypeError(
                        f"Unknown server params type for {server_name}: {type(server_params)}"
                    )

                logger.info(f"Tool '{tool_name}' (server: '{server_name}') called successfully.")

                if isinstance(result_content, str) and "Unknown tool:" in result_content:
                    suggested_servers = await self._find_servers_with_tool(tool_name)
                    if len(suggested_servers) == 1:
                        logger.info(
                            f"Auto-correction: Tool '{tool_name}' not found in '{server_name}', trying '{suggested_servers[0]}'"
                        )
                        return await self.execute_tool_call(
                            suggested_servers[0], tool_name, arguments
                        )

                return {
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "result": result_content,  # Return extracted text content
                }

            except Exception as outer_e:  # Rename this to outer_e to avoid shadowing
                logger.error(
                    f"Error: Failed to call tool '{tool_name}' (server: '{server_name}'): {outer_e}"
                )
                # import traceback
                # traceback.print_exc() # Print detailed stack trace for debugging

                # Store the original error message for later use
                error_message = str(outer_e)

                if (
                    tool_name == "scrape"
                    and "unhandled errors" in error_message
                    and "url" in arguments
                    and arguments["url"] is not None
                ):
                    try:
                        logger.info("Attempting to use MarkItDown for fallback...")
                        from markitdown import MarkItDown

                        md = MarkItDown(docintel_endpoint="<document_intelligence_endpoint>")
                        result = md.convert(arguments["url"])
                        logger.info("Successfully used MarkItDown")
                        return {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "result": result.text_content,  # Return extracted text content
                        }
                    except Exception as inner_e:  # Use a different name to avoid shadowing
                        # Log the inner exception if needed
                        logger.error(f"Fallback also failed: {inner_e}")
                        # No need for pass here as we'll continue to the return statement

                # Always use the outer exception for the final error response
                return {
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "error": f"Tool call failed: {error_message}",
                }
