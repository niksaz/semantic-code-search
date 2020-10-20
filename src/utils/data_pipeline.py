import collections
from typing import Optional

from dpu_utils.utils import RichPath

CODE_TOKENS_LABEL = 'code_tokens'
RAW_TREE_LABEL = 'raw_tree'

TreeNode = collections.OrderedDict


def _get_child_with_type(node: TreeNode, type_str: str) -> Optional[int]:
  for index, child in enumerate(node['children']):
    if child['type'] == type_str:
      return index
  return None


def _remove_docstring_node(root: TreeNode) -> None:
  """The docstring node follows the structure of:
  module
    function_definition
      ...
      block
        expression_statement
          string
  """
  try:
    assert root['type'] == 'module'
    function_definition_index = _get_child_with_type(root, 'function_definition')
    assert function_definition_index == 0
    function_definition_node = root['children'][function_definition_index]
    block_index = _get_child_with_type(function_definition_node, 'block')
    assert block_index is not None
    block_node = function_definition_node['children'][block_index]
    expression_statement_index = _get_child_with_type(block_node, 'expression_statement')
    assert expression_statement_index == 0
    expression_statement_node = block_node['children'][expression_statement_index]
    string_index = _get_child_with_type(expression_statement_node, 'string')
    assert string_index == 0
    # Remove the expression with the string node which corresponds to the docstring.
    block_node['children'].pop(expression_statement_index)
  except AssertionError:
    pass


def _original_to_raw_tree_path(file_path: RichPath, language: str):
  raw_tree_path = file_path.__str__().replace(f'/{language}/', f'/{language}_raw_trees/')
  return RichPath.create(raw_tree_path)


def combined_samples_generator(data_file: RichPath):
  raw_tree_iterator = None
  for raw_sample in data_file.read_by_file_suffix():
    assert CODE_TOKENS_LABEL in raw_sample
    if raw_tree_iterator is None:
      raw_tree_path = _original_to_raw_tree_path(data_file, language=raw_sample['language'])
      raw_tree_iterator = raw_tree_path.read_by_file_suffix()
    raw_tree = next(raw_tree_iterator)
    _remove_docstring_node(raw_tree)
    assert RAW_TREE_LABEL not in raw_sample
    raw_sample[RAW_TREE_LABEL] = raw_tree
    yield raw_sample
