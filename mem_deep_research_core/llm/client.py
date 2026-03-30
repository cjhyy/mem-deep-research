import importlib

from omegaconf import DictConfig, OmegaConf

from mem_deep_research_core.exceptions import ConfigValidationError, LLMProviderNotFoundError
from mem_deep_research_core.mem_deep_research_logging.task_tracer import TaskTracer


def LLMClient(
    task_id: str,
    cfg: DictConfig = None,
    llm_config: DictConfig = None,
    task_log: TaskTracer | None = None,
    **kwargs,
):
    """
    create LLMClientProvider from hydra configuration.
    Can accept either:
    - cfg: Traditional config with cfg.llm structure
    - llm_config: Direct LLM configuration
    """
    if llm_config is not None:
        # Direct LLM config provided
        provider_class = llm_config.provider_class
        # Create compatible config structure
        config = OmegaConf.create({"llm": llm_config})
        config = OmegaConf.merge(config, kwargs)
    elif cfg is not None:
        # Traditional cfg.llm structure
        provider_class = cfg.llm.provider_class
        config = OmegaConf.merge(cfg, kwargs)
    else:
        raise ValueError("Either cfg or llm_config must be provided")

    # Validate config type (replaced assert with explicit check)
    if not isinstance(config, DictConfig):
        raise ConfigValidationError(
            f"Expected DictConfig, got {type(config).__name__}",
            field="config",
            value=type(config).__name__,
        )

    # Dynamically import the provider class from the .providers module

    # Validate provider_class is a string and a valid identifier
    if not isinstance(provider_class, str) or not provider_class.isidentifier():
        raise ConfigValidationError(
            "Invalid provider_class: must be a valid Python identifier",
            field="provider_class",
            value=provider_class,
        )

    try:
        # Import the module dynamically
        providers_module = importlib.import_module("mem_deep_research_core.llm.providers")
        # Get the class from the module
        ProviderClass = getattr(providers_module, provider_class)
    except ModuleNotFoundError as e:
        raise LLMProviderNotFoundError(provider_class) from e
    except AttributeError as e:
        raise LLMProviderNotFoundError(provider_class) from e

    # Instantiate the client using the imported class
    try:
        client_instance = ProviderClass(task_id=task_id, task_log=task_log, cfg=config)
    except (TypeError, ValueError) as e:
        raise ConfigValidationError(
            f"Failed to instantiate {provider_class}: {e}",
            field="provider_class",
            value=provider_class,
        ) from e

    return client_instance
