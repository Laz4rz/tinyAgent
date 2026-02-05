from __future__ import annotations

import inspect
from typing import Any, get_origin, get_args


_TYPE_MAP = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
}


def make_tool_schema(fn) -> dict[str, Any]:
    """Create a minimal JSON-schema-like tool description from a function."""
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or ""
    props: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            # *args -> array
            item_type = _type_to_schema(param.annotation).get("items", {"type": "string"})
            props[name] = {"type": "array", "items": item_type, "description": "varargs"}
            continue

        schema = _type_to_schema(param.annotation)
        if param.default is inspect._empty:
            required.append(name)
        else:
            schema["default"] = param.default
        props[name] = schema

    return {
        "name": fn.__name__,
        "description": doc.splitlines()[0] if doc else "",
        "parameters": {
            "type": "object",
            "properties": props,
            "required": required,
            "additionalProperties": False,
        },
    }


def validate_and_call(fn, args: dict[str, Any]) -> Any:
    """Validate args using the function signature and basic types, then call."""
    sig = inspect.signature(fn)
    bound = sig.bind_partial()

    for name, param in sig.parameters.items():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            varargs = args.get(name, [])
            if not isinstance(varargs, list):
                raise TypeError(f"{name} must be a list")
            for item in varargs:
                _check_type(name, item, param.annotation)
            bound.arguments[name] = varargs
            continue

        if name not in args:
            if param.default is inspect._empty:
                raise TypeError(f"missing required arg: {name}")
            bound.arguments[name] = param.default
            continue

        value = args[name]
        _check_type(name, value, param.annotation)
        bound.arguments[name] = value

    if any(k not in sig.parameters for k in args):
        extra = [k for k in args if k not in sig.parameters]
        raise TypeError(f"unexpected args: {', '.join(extra)}")

    return fn(*bound.args, **bound.kwargs)


def _type_to_schema(tp) -> dict[str, Any]:
    if tp is inspect._empty:
        return {"type": "string"}
    origin = get_origin(tp)
    if origin is list:
        args = get_args(tp)
        item = _type_to_schema(args[0] if args else str)
        return {"type": "array", "items": item}
    return {"type": _TYPE_MAP.get(tp, "string")}


def _check_type(name: str, value: Any, tp) -> None:
    if tp is inspect._empty:
        return
    origin = get_origin(tp)
    if origin is list:
        if not isinstance(value, list):
            raise TypeError(f"{name} must be a list")
        return
    if tp in _TYPE_MAP and not isinstance(value, tp):
        raise TypeError(f"{name} must be {tp.__name__}")

