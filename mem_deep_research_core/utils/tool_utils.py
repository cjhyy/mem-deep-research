import contextlib
import os
import pathlib
import sys

from mcp import StdioServerParameters
from omegaconf import DictConfig, OmegaConf

from mem_deep_research_core.core.constants import SUB_AGENT_PREFIX
from mem_deep_research_core.mem_deep_research_logging.logger import bootstrap_logger
from mem_deep_research_core.prompts import AgentPrompt
from mem_deep_research_core.utils.external_loader import ConfigLoader

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
logger = bootstrap_logger(level=LOGGER_LEVEL)


# MCP server configuration generation function
def create_mcp_server_parameters(
    cfg: DictConfig, agent_cfg: DictConfig, logs_dir: str | None = None, config_loader=None
):
    """Define and return MCP server configuration list

    支持四种工具配置模式：
    1. stdio 模式（本地）: 启动本地子进程
    2. http 模式（远程）: 连接远程 MCP 服务器 URL
    3. sse 模式（远程）: 连接远程 SSE 端点
    4. inprocess 模式（进程内）: 直接导入 Python 模块，无需子进程

    配置示例：
    ```yaml
    # stdio 模式（本地）
    name: "tool-local"
    tool_command: "python"
    args: ["my_server.py"]

    # http 模式（远程）
    name: "tool-remote"
    url: "http://localhost:8080/mcp"

    # sse 模式（远程）
    name: "tool-remote-sse"
    url: "http://localhost:8080/sse"
    transport: "sse"

    # inprocess 模式（进程内，无子进程开销）
    name: "tool-calculator"
    transport: "inprocess"
    module: "mem_deep_research_core.tool.mcp_servers.calculator_server"
    object: "mcp"
    ```
    """
    if config_loader is None:
        raise ValueError("config_loader is required for create_mcp_server_parameters")
    _loader = config_loader
    configs = []

    if agent_cfg.get("tool_config", None) is not None:
        for tool in agent_cfg["tool_config"]:
            try:
                # 优先从外部加载工具配置
                tool_cfg_resolved = None
                with contextlib.suppress(ValueError):
                    tool_cfg_resolved = _loader.load_tool_config(tool)

                # 回退到框架内置配置
                if tool_cfg_resolved is None:
                    config_path = (
                        pathlib.Path(__file__).parent.parent.parent
                        / "config"
                        / "tool"
                        / f"{tool}.yaml"
                    )
                    tool_cfg = OmegaConf.load(config_path)
                    # Resolve OmegaConf interpolations (e.g., ${oc.env:VAR,default})
                    tool_cfg_resolved = OmegaConf.to_container(tool_cfg, resolve=True)

                tool_name = tool_cfg_resolved.get("name", tool)

                # 判断是远程 URL 还是本地命令
                if "url" in tool_cfg_resolved:
                    # 远程 MCP 服务器（HTTP 或 SSE 模式）
                    url = tool_cfg_resolved["url"]
                    transport = tool_cfg_resolved.get("transport", "streamable-http")
                    headers = tool_cfg_resolved.get("headers", {})

                    inject_context = tool_cfg_resolved.get("inject_context", True)

                    configs.append(
                        {
                            "name": tool_name,
                            "params": url,  # 直接使用 URL 字符串
                            "transport": transport,
                            "headers": headers,
                            "inject_context": inject_context,
                        }
                    )
                    logger.info(
                        f"[ToolUtils] Configured remote MCP server '{tool_name}': {url} (transport={transport})"
                    )
                elif tool_cfg_resolved.get("transport") == "inprocess":
                    # In-process MCP tool — no subprocess, direct Python import
                    configs.append(
                        {
                            "name": tool_name,
                            "params": "inprocess",  # Sentinel: truthy, distinguishes from None
                            "transport": "inprocess",
                            "module": tool_cfg_resolved.get("module", ""),
                            "object": tool_cfg_resolved.get("object", "mcp"),
                        }
                    )
                    logger.info(
                        f"[ToolUtils] Configured in-process MCP tool '{tool_name}': "
                        f"module={tool_cfg_resolved.get('module')}"
                    )
                else:
                    # 本地 stdio 模式
                    env_dict = tool_cfg_resolved.get("env", {}) or {}
                    env_str_dict = {k: str(v) if v is not None else "" for k, v in env_dict.items()}

                    configs.append(
                        {
                            "name": tool_name,
                            "params": StdioServerParameters(
                                command=sys.executable
                                if tool_cfg_resolved["tool_command"] == "python"
                                else tool_cfg_resolved["tool_command"],
                                args=tool_cfg_resolved.get("args", []),
                                env=env_str_dict,
                            ),
                        }
                    )
            except Exception as e:
                logger.error(f"[ERROR] Error creating MCP server parameters for tool {tool}: {e}")
                continue

    blacklist = set()
    for black_list_item in agent_cfg.get("tool_blacklist", []):
        if isinstance(black_list_item, (list, tuple)) and len(black_list_item) >= 2:
            blacklist.add((black_list_item[0], black_list_item[1]))
        else:
            logger.warning(
                f"[ToolUtils] Invalid blacklist entry (expected [server, tool]): {black_list_item}"
            )
    return configs, blacklist


def _load_agent_prompt(prompt_cfg: dict = None) -> AgentPrompt:
    """加载 Agent 提示词

    Args:
        prompt_cfg: Prompt 配置字典
            - agent_type: "main" 或 "worker"
            - tool_format: "xml" 或 "native"
            - presets: 预设列表，如 ["research", "time_sensitive"]
            - templates_dir: 自定义模板目录
            - custom_system_template: 自定义系统模板
            - custom_summarize_template: 自定义总结模板

    Returns:
        AgentPrompt 实例
    """
    if not prompt_cfg:
        prompt_cfg = {}

    return AgentPrompt(
        agent_type=prompt_cfg.get("agent_type", "main"),
        tool_format=prompt_cfg.get("tool_format", "xml"),
        presets=prompt_cfg.get("presets", []),
        templates_dir=prompt_cfg.get("templates_dir"),
        custom_system_template=prompt_cfg.get("custom_system_template"),
        custom_summarize_template=prompt_cfg.get("custom_summarize_template"),
        minimal=prompt_cfg.get("minimal", False),
    )


def expose_sub_agents_as_tools(sub_agents_cfg: DictConfig):
    """Expose sub-agents as tools"""
    sub_agents_server_params = []
    for sub_agent in sub_agents_cfg:
        if not sub_agent.startswith(SUB_AGENT_PREFIX):
            raise ValueError(
                f"Sub-agent name must start with '{SUB_AGENT_PREFIX}': {sub_agent}. Please check the sub-agent name in the agent's config file."
            )
        try:
            sub_agent_cfg = sub_agents_cfg[sub_agent]

            # 获取 prompt 配置
            prompt_cfg = {}
            if hasattr(sub_agent_cfg, "prompt") and sub_agent_cfg.prompt:
                prompt_cfg = dict(sub_agent_cfg.prompt)
            # 子 Agent 默认为 worker 类型
            if "agent_type" not in prompt_cfg:
                prompt_cfg["agent_type"] = "worker"

            sub_agent_prompt_instance = _load_agent_prompt(prompt_cfg)
            sub_agent_tool_definition = sub_agent_prompt_instance.expose_agent_as_tool(
                subagent_name=sub_agent
            )
            sub_agents_server_params.append(sub_agent_tool_definition)
        except Exception as e:
            raise ValueError(f"Failed to expose sub-agent {sub_agent} as a tool: {e}")
    return sub_agents_server_params
