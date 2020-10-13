import collections
import re
from typing import List

TreeNode = collections.OrderedDict


def _linearize_tree(node: TreeNode, linearization: List[TreeNode]):
  linearization.append(node)
  for child in node['children']:
    _linearize_tree(child, linearization)


def _get_code_tokens_from_tree(tree: collections.OrderedDict):
  linearization = []
  _linearize_tree(tree, linearization)
  node_tokens = list(map(lambda node: node['string'], linearization))
  python_identifier_pattern = re.compile(r'^[^\d\W]\w*\Z', re.UNICODE)
  code_tokens = list(filter(lambda token: re.match(python_identifier_pattern, token), node_tokens))
  return code_tokens


def _get_types_bag_from_tree(tree: collections.OrderedDict):
  linearization = []
  _linearize_tree(tree, linearization)
  type_tokens = list(map(lambda node: node['type'], linearization))
  types_bag = set(type_tokens)
  return types_bag


def mix_raw_tree_in(raw_sample, raw_tree):
  raw_sample['raw_tree'] = raw_tree
  return raw_sample
