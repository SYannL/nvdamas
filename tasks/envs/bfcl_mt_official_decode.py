"""
与 BFCL `default_decode_execute_prompting` + `ast_parse`(PYTHON) 等价逻辑，
单独放在此文件以避免 `import bfcl_eval.model_handler.utils` 时拉取 java_parser / tree_sitter。

逻辑对齐：berkeley-function-call-leaderboard/bfcl_eval/model_handler/utils.py
（resolve_ast_* / parse_nested_value / decoded_output_to_execution_list / default_decode_execute_prompting）。
"""

from __future__ import annotations

import ast


def resolve_ast_call(elem: ast.Call) -> dict:
    func_parts: list[str] = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        args_dict[arg.arg] = resolve_ast_by_type(arg.value)
    return {func_name: args_dict}


def resolve_ast_by_type(value: ast.AST):
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            return "..."
        return value.value
    if isinstance(value, ast.UnaryOp):
        return -value.operand.value  # type: ignore[operator]
    if isinstance(value, ast.List):
        return [resolve_ast_by_type(v) for v in value.elts]
    if isinstance(value, ast.Dict):
        return {
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    if isinstance(value, ast.NameConstant):
        return value.value
    if isinstance(value, ast.BinOp):
        return eval(ast.unparse(value))
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            return ast.unparse(value)
        return resolve_ast_call(value)
    if isinstance(value, ast.Tuple):
        return tuple(resolve_ast_by_type(v) for v in value.elts)
    if isinstance(value, ast.Lambda):
        return eval(ast.unparse(value.body[0].value))  # type: ignore[index]
    if isinstance(value, ast.Ellipsis):
        return "..."
    if isinstance(value, ast.Subscript):
        try:
            return ast.unparse(value.body[0].value)  # type: ignore[index]
        except Exception:
            return ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    raise Exception(f"Unsupported AST type: {type(value)}")


def parse_nested_value(value) -> str:
    if isinstance(value, dict):
        if all(isinstance(v, dict) for v in value.values()):
            func_name = list(value.keys())[0]
            args = value[func_name]
            args_str = ", ".join(f"{k}={parse_nested_value(v)}" for k, v in args.items())
            return f"{func_name}({args_str})"
        return "{" + ", ".join(f"'{k}': {parse_nested_value(v)}" for k, v in value.items()) + "}"
    return repr(value)


def decoded_output_to_execution_list(decoded_output: list[dict]) -> list[str]:
    execution_list: list[str] = []
    for function_call in decoded_output:
        for key, value in function_call.items():
            args_str = ", ".join(f"{k}={parse_nested_value(v)}" for k, v in value.items())
            execution_list.append(f"{key}({args_str})")
    return execution_list


def ast_parse_python_calls(input_str: str) -> list[dict]:
    cleaned_input = input_str.strip().strip("'")
    parsed = ast.parse(cleaned_input, mode="eval")
    root = parsed.body
    extracted: list[dict] = []
    if isinstance(root, ast.Call):
        extracted.append(resolve_ast_call(root))
    elif isinstance(root, (ast.List, ast.Tuple)):
        for elem in root.elts:
            if not isinstance(elem, ast.Call):
                raise ValueError(f"BFCL decode: list element must be Call, got {type(elem)}")
            extracted.append(resolve_ast_call(elem))
    else:
        raise ValueError(f"BFCL decode: unsupported expression {type(root)}")
    return extracted


def decode_python_fc_calls(result: str) -> list[str]:
    """等价于 bfcl_eval.model_handler.utils.default_decode_execute_prompting（仅 Python）。"""
    result = result.strip("`\n ")
    if not result.startswith("["):
        result = "[" + result
    if not result.endswith("]"):
        result = result + "]"
    decoded_output = ast_parse_python_calls(result)
    return decoded_output_to_execution_list(decoded_output)
