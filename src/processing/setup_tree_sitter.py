from tree_sitter import Language

from processing.utils import tree_sitter_so, tree_sitter_languages

Language.build_library(
  # Store the library in the `build` directory
  tree_sitter_so(),

  # Include one or more languages
  [
    tree_sitter_languages('tree-sitter-python')
  ]
)
