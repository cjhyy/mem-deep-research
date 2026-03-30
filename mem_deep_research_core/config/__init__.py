# This file makes the conf directory a Python package
import logging
import pathlib

# 延迟导入，避免循环依赖
try:
    import omegaconf
    import yaml
except ImportError:
    omegaconf = None
    yaml = None


def config_path() -> str:
    return str(pathlib.Path(__file__).parent.absolute())


def config_name() -> str:
    return "config"


def debug_config(cfg, logger: logging.Logger):
    """调试配置 - 延迟导入依赖"""
    import yaml

    try:
        import omegaconf

        full_config = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        # 如果不是 OmegaConf 对象，直接使用
        full_config = cfg
    logger.info(yaml.dump(data=full_config))


__all__ = [
    "config_path",
    "config_name",
    "debug_config",
]
