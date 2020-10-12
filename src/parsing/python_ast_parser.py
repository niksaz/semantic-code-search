from typing import Dict, Any
from tree_sitter.binding import Parser, Node

from parsing.code_parser import CodeParser
from parsing.utils import PY_LANGUAGE


class PythonAstParser(CodeParser):

    def __init__(self):
        super().__init__()
        self.parser = Parser()
        self.parser.set_language(PY_LANGUAGE)

    def parse_function(self, code: str) -> Dict[str, Any]:
        bytecode = bytes(code, "utf8")
        root = self.parser.parse(bytecode).root_node

        def get_span(node: Node) -> str:
            return str(bytecode[node.start_byte:node.end_byte], "utf8")

        def build_json_tree(node: Node) -> Dict[str, Any]:
            return {
                "type": node.type,
                "string": get_span(node) if node.type == "string" or len(node.children) == 0 else "",
                "children": [build_json_tree(child) for child in node.children]
            }

        return build_json_tree(root)
