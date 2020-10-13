import collections
import re
from typing import List

from dpu_utils.utils import RichPath


def query_to_raw_tree_path(file_path: RichPath):
  raw_tree_path = file_path.__str__().replace('/python/', '/python_raw_trees/')
  return RichPath.create(raw_tree_path)


def _add_code_tokens_from_node(node: collections.OrderedDict, tokens: List[str]):
  node_string = node['string']
  python_identifier = re.compile(r'^[^\d\W]\w*\Z', re.UNICODE)
  if re.match(python_identifier, node_string):
    tokens.append(node_string)
  for child in node['children']:
    _add_code_tokens_from_node(child, tokens)


def get_code_tokens_from_tree(tree: collections.OrderedDict):
  code_tokens = []
  _add_code_tokens_from_node(tree, code_tokens)
  return code_tokens


def mix_raw_tree_in(raw_sample, raw_tree):
  raw_sample['code_tokens'] = get_code_tokens_from_tree(raw_tree)
  return raw_sample
