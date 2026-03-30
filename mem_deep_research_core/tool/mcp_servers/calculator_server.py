"""
简单计算器 MCP 工具示例

演示如何创建自定义 MCP 工具供框架调用。
"""

import ast
import math
import operator

from fastmcp import FastMCP

# 初始化 MCP 服务器
mcp = FastMCP("calculator-server")

# AST-safe 数学表达式求值器（替代 eval 以防止代码注入）
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log10,
    "ln": math.log,
    "exp": math.exp,
    "abs": abs,
    "pow": pow,
    "round": round,
    "min": min,
    "max": max,
}

_SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval_node(node: ast.AST) -> float | int:
    """Recursively evaluate an AST node using only whitelisted operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _SAFE_CONSTANTS:
        return _SAFE_CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](
            _safe_eval_node(node.left), _safe_eval_node(node.right)
        )
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCTIONS:
            args = [_safe_eval_node(arg) for arg in node.args]
            return _SAFE_FUNCTIONS[node.func.id](*args)
        raise ValueError(f"不允许的函数调用: {ast.dump(node.func)}")
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


def safe_eval_expression(expression: str) -> float | int:
    """Parse and evaluate a math expression safely via AST."""
    tree = ast.parse(expression, mode="eval")
    return _safe_eval_node(tree)


@mcp.tool()
async def calculate(expression: str) -> str:
    """执行数学计算表达式。

    支持基本运算 (+, -, *, /, //, %, **) 和数学函数 (sqrt, sin, cos, log 等)。

    Args:
        expression: 数学表达式，如 "2 + 3 * 4" 或 "sqrt(16) + log(100)"

    Returns:
        计算结果
    """
    try:
        result = safe_eval_expression(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"[ERROR]: 计算失败 - {str(e)}"


@mcp.tool()
async def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """单位转换工具。

    Args:
        value: 要转换的数值
        from_unit: 原单位 (km, m, cm, mm, mile, ft, inch, kg, g, lb, oz, c, f, k)
        to_unit: 目标单位

    Returns:
        转换结果
    """
    # 长度单位（以米为基准）
    length_units = {
        "km": 1000,
        "m": 1,
        "cm": 0.01,
        "mm": 0.001,
        "mile": 1609.344,
        "ft": 0.3048,
        "inch": 0.0254,
    }

    # 重量单位（以克为基准）
    weight_units = {"kg": 1000, "g": 1, "mg": 0.001, "lb": 453.592, "oz": 28.3495}

    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    try:
        # 温度转换
        if from_unit in ["c", "f", "k"] and to_unit in ["c", "f", "k"]:
            if from_unit == "c":
                celsius = value
            elif from_unit == "f":
                celsius = (value - 32) * 5 / 9
            else:  # k
                celsius = value - 273.15

            if to_unit == "c":
                result = celsius
            elif to_unit == "f":
                result = celsius * 9 / 5 + 32
            else:  # k
                result = celsius + 273.15

            return f"{value} {from_unit.upper()} = {result:.2f} {to_unit.upper()}"

        # 长度转换
        if from_unit in length_units and to_unit in length_units:
            meters = value * length_units[from_unit]
            result = meters / length_units[to_unit]
            return f"{value} {from_unit} = {result:.4f} {to_unit}"

        # 重量转换
        if from_unit in weight_units and to_unit in weight_units:
            grams = value * weight_units[from_unit]
            result = grams / weight_units[to_unit]
            return f"{value} {from_unit} = {result:.4f} {to_unit}"

        return f"[ERROR]: 不支持的单位转换: {from_unit} -> {to_unit}"
    except Exception as e:
        return f"[ERROR]: 转换失败 - {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
