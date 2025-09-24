from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

@dataclass(frozen=True)
class CIFRecord:
    cod_id: str
    path: str
    """ ... """


@dataclass(frozen=True)
class HKLRecord:
    cod_id: str
    path: str
    """ ... """


class Node:
    __slots__ = ("children", "records", "is_terminal")
    def __init__(self):
        self.children: Dict[str, Node] = {}
        # can hold CIFRecord or HKLRecord
        self.records: List[object] = []
        self.is_terminal: bool = False


class Trie:
    def __init__(self):
        self.root = Node()


class BuildTrie:
    pass