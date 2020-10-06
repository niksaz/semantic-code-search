import os

current_folder = os.path.dirname(__file__)


def tree_sitter_so() -> str:
    return os.path.join(current_folder, 'build', 'my-languages.so')


def tree_sitter_languages(language: str) -> str:
    return os.path.join(current_folder, 'tree-sitter-languages', language)
