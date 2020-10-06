from typing import Dict, Any

from tree_sitter import Language
from tree_sitter.binding import Parser, Node

from processing.code_parser import CodeParser
from processing.utils import tree_sitter_so

GO_LANGUAGE = Language(tree_sitter_so(), 'go')
JAVA_LANGUAGE = Language(tree_sitter_so(), 'java')
JS_LANGUAGE = Language(tree_sitter_so(), 'javascript')
PHP_LANGUAGE = Language(tree_sitter_so(), 'php')
PY_LANGUAGE = Language(tree_sitter_so(), 'python')
RUBY_LANGUAGE = Language(tree_sitter_so(), 'ruby')


class TreeSitterAstParser(CodeParser):

    def __init__(self, language: Language):
        super().__init__()
        self.parser = Parser()
        self.parser.set_language(language)

    def process_data(self, code: str) -> Dict[str, Any]:
        root = self.parser.parse(bytes(code, "utf8")).root_node

        def get_span(node: Node) -> str:
            return code[node.start_byte:node.end_byte]

        def build_json_tree(node: Node) -> Dict[str, Any]:
            return {
                "type": node.type,
                "string": get_span(node) if node.type == "string" or len(node.children) == 0 else "",
                "children": [build_json_tree(child) for child in node.children]
            }

        return build_json_tree(root)


class GoAstParser(TreeSitterAstParser):
    def __init__(self):
        super().__init__(GO_LANGUAGE)


class JavaAstParser(TreeSitterAstParser):
    def __init__(self):
        super().__init__(JAVA_LANGUAGE)


class JavascriptAstParser(TreeSitterAstParser):
    def __init__(self):
        super().__init__(JS_LANGUAGE)


class PhpAstParser(TreeSitterAstParser):
    def __init__(self):
        super().__init__(PHP_LANGUAGE)


class PythonAstParser(TreeSitterAstParser):
    def __init__(self):
        super().__init__(PY_LANGUAGE)


class RubyAstParser(TreeSitterAstParser):
    def __init__(self):
        super().__init__(RUBY_LANGUAGE)
