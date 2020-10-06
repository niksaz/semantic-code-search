import os

from tree_sitter import Language

current_folder = os.path.dirname(__file__)


def tree_sitter_so() -> str:
    return os.path.join(current_folder, 'build', 'my-languages.so')


def tree_sitter_languages(language: str) -> str:
    return os.path.join(current_folder, 'tree-sitter-languages', language)


GO_LANGUAGE = Language(tree_sitter_so(), 'go')
JAVA_LANGUAGE = Language(tree_sitter_so(), 'java')
JS_LANGUAGE = Language(tree_sitter_so(), 'javascript')
PHP_LANGUAGE = Language(tree_sitter_so(), 'php')
PY_LANGUAGE = Language(tree_sitter_so(), 'python')
RUBY_LANGUAGE = Language(tree_sitter_so(), 'ruby')
