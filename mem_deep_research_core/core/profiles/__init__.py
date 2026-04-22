"""Profile 系统

Profile 决定 agent 的执行策略（research / standard / automation / ...）。
与 Mode（quick / standard / deep，资源等级）正交。

使用：
    # 字符串方式（内置 profile）
    from mem_deep_research_core.core.profiles import resolve_profile
    profile = resolve_profile("standard")

    # 对象方式
    from mem_deep_research_core.core.profiles import StandardProfile
    profile = StandardProfile()

注册自定义 profile：
    from mem_deep_research_core.core.profiles import register_profile

    class MyProfile(Profile):
        name = "my_profile"
        ...

    register_profile(MyProfile)
"""

from mem_deep_research_core.core.profiles.base import Profile, ProfileContext
from mem_deep_research_core.core.profiles.deep_research import DeepResearchProfile
from mem_deep_research_core.core.profiles.standard import StandardProfile

# Name → Profile class 注册表
_PROFILE_REGISTRY: dict[str, type[Profile]] = {
    "standard": StandardProfile,
    "deep_research": DeepResearchProfile,
}


def register_profile(profile_cls: type[Profile]) -> None:
    """注册自定义 Profile 类到全局 registry。

    Profile 类必须设置 name 属性。重名会覆盖。
    """
    if not issubclass(profile_cls, Profile):
        raise TypeError(f"{profile_cls!r} must be a Profile subclass")
    name = getattr(profile_cls, "name", None)
    if not name or name == "base":
        raise ValueError(
            f"Profile class {profile_cls!r} must define a non-empty 'name' attribute"
        )
    _PROFILE_REGISTRY[name] = profile_cls


def resolve_profile(
    profile: str | Profile | type[Profile] | None,
    config: dict | None = None,
) -> Profile:
    """将 profile 参数解析为 Profile 实例。

    接受：
    - None → StandardProfile()
    - 字符串 → 从 registry 查找对应 class 并实例化
    - Profile class → 实例化
    - Profile instance → 直接返回

    Args:
        profile: profile 标识或对象
        config: profile 配置，仅在需要新建实例时使用

    Returns:
        Profile 实例

    Raises:
        ValueError: 字符串名不在 registry 中
        TypeError: 传入类型非法
    """
    if profile is None:
        return StandardProfile()

    if isinstance(profile, Profile):
        return profile

    if isinstance(profile, type) and issubclass(profile, Profile):
        return _build_from_class(profile, config)

    if isinstance(profile, str):
        if profile not in _PROFILE_REGISTRY:
            available = sorted(_PROFILE_REGISTRY.keys())
            raise ValueError(
                f"Unknown profile name {profile!r}. Available: {available}"
            )
        return _build_from_class(_PROFILE_REGISTRY[profile], config)

    raise TypeError(
        f"profile must be str, Profile instance, or Profile subclass; got {type(profile)!r}"
    )


def _build_from_class(cls: type[Profile], config: dict | None) -> Profile:
    """尝试用 config 构造 profile，失败时无参构造（兼容 StandardProfile）。"""
    if config:
        try:
            return cls(config)  # type: ignore[call-arg]
        except TypeError:
            # Profile 可能是无参构造的（如 StandardProfile），config 忽略
            inst = cls()
            return inst
    return cls()


def list_profiles() -> list[str]:
    """返回所有已注册的 profile 名称（按名称排序）。"""
    return sorted(_PROFILE_REGISTRY.keys())


__all__ = [
    "Profile",
    "ProfileContext",
    "StandardProfile",
    "DeepResearchProfile",
    "register_profile",
    "resolve_profile",
    "list_profiles",
]
