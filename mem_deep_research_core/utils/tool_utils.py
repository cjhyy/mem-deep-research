import contextlib
import os
import pathlib
import sys

from mcp import StdioServerParameters
from omegaconf import DictConfig, OmegaConf

from mem_deep_research_core.mem_deep_research_logging.logger import bootstrap_logger
from mem_deep_research_core.prompts import AgentPrompt
from mem_deep_research_core.utils.external_loader import external_loader

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
logger = bootstrap_logger(level=LOGGER_LEVEL)


# MCP server configuration generation function
def create_mcp_server_parameters(
    cfg: DictConfig, agent_cfg: DictConfig, logs_dir: str | None = None
):
    """Define and return MCP server configuration list

    支持三种工具配置模式：
    1. stdio 模式（本地）: 启动本地子进程
    2. http 模式（远程）: 连接远程 MCP 服务器 URL
    3. sse 模式（远程）: 连接远程 SSE 端点

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
    ```
    """
    configs = []

    if agent_cfg.get("tool_config", None) is not None:
        for tool in agent_cfg["tool_config"]:
            try:
                # 优先从外部加载工具配置
                tool_cfg_resolved = None
                with contextlib.suppress(ValueError):
                    tool_cfg_resolved = external_loader.load_tool_config(tool)

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
        blacklist.add((black_list_item[0], black_list_item[1]))
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
    )


def expose_sub_agents_as_tools(sub_agents_cfg: DictConfig):
    """Expose sub-agents as tools"""
    sub_agents_server_params = []
    for sub_agent in sub_agents_cfg:
        if not sub_agent.startswith("agent-"):
            raise ValueError(
                f"Sub-agent name must start with 'agent-': {sub_agent}. Please check the sub-agent name in the agent's config file."
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
