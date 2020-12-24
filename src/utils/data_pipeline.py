import collections
from typing import Dict, Optional

from dpu_utils.utils import RichPath

CODE_TOKENS_LABEL = 'code_tokens'
TREE_LABEL = '_raw_trees'
COMPRESSED_TREE_LABEL = '_compressed_100'
GRAPH_LABEL = '_graphs'

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


def combined_samples_generator(resource_mapping: Dict[str, Optional[RichPath]]):
  language = 'python'
  # If we are training, load all available resources.
  if len(resource_mapping) == 1 and list(resource_mapping.keys())[0] == CODE_TOKENS_LABEL:
    code_data_file = resource_mapping.get(CODE_TOKENS_LABEL)
    assert code_data_file
    for resource in [TREE_LABEL, COMPRESSED_TREE_LABEL, GRAPH_LABEL]:
      resource_path = str(code_data_file).replace(f'/{language}/', f'/{language}{resource}/')
      data_file = RichPath.create(resource_path)
      resource_mapping[resource] = data_file

  resource_iterator = {}
  for resource, data_file in resource_mapping.items():
    assert resource in [CODE_TOKENS_LABEL, TREE_LABEL, COMPRESSED_TREE_LABEL, GRAPH_LABEL]
    resource_iterator[resource] = data_file.read_by_file_suffix()

  while True:
    sample = {}
    try:
      for resource, iterator in resource_iterator.items():
        resource_item = next(iterator)
        if resource == CODE_TOKENS_LABEL:
          assert CODE_TOKENS_LABEL in resource_item
          sample.update(resource_item)
        elif resource in [TREE_LABEL, COMPRESSED_TREE_LABEL]:
          _remove_docstring_node(resource_item)
          sample[resource] = resource_item
        elif resource == GRAPH_LABEL:
          sample[resource] = resource_item
        else:
          raise ValueError(f'Generator for {resource} is not implemented')
    except StopIteration:
      break
    yield sample
