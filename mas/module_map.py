from typing import Type

from .reasoning import ReasoningBase, ReasoningIO
from .memory import (
    MASMemoryBase,
    VoyagerMASMemory,
    MemoryBankMASMemory,
    MemoryBankGraphMASMemory,
    ChatDevMASMemory,
    GenerativeMASMemory,
    MetaGPTMASMemory,
    GMemory,
    GMemoryGraphMASMemory,
    SelectiveMemMASMemory,
    GraphMemory2MASMemory,
    GraphMemory3MASMemory,
    MemRLMASMemory,
)


def module_map(
    reasoning: str, mas_memory: str | None = None
) -> tuple[Type[ReasoningBase], Type[MASMemoryBase]]:
    reasoning_map = {
        'io': ReasoningIO,
    }
    mas_memory_map: dict[str, Type[MASMemoryBase]] = {
        'empty': MASMemoryBase,
        'voyager': VoyagerMASMemory,
        'memorybank': MemoryBankMASMemory,
        'memgraph': MemoryBankGraphMASMemory,
        'chatdev': ChatDevMASMemory,
        'generative': GenerativeMASMemory,
        'metagpt': MetaGPTMASMemory,
        'g-memory': GMemory,
        'gmemgraph': GMemoryGraphMASMemory,
        'selectivemem': SelectiveMemMASMemory,
        'graph_memory2': GraphMemory2MASMemory,
        'graph_memory3': GraphMemory3MASMemory,
        'memrl': MemRLMASMemory,
    }

    if reasoning not in reasoning_map:
        raise ValueError(f"Invalid reasoning type '{reasoning}'. Allowed values: {list(reasoning_map.keys())}")

    if mas_memory is not None and mas_memory not in mas_memory_map:
        raise ValueError(f"Invalid MAS memory type '{mas_memory}'. Allowed values: {list(mas_memory_map.keys())}")

    return (
        reasoning_map[reasoning],
        mas_memory_map.get(mas_memory, MASMemoryBase),
    )
