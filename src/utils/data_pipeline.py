import collections
import numpy as np
from typing import Optional, Dict, Any

from dpu_utils.utils import RichPath

CODE_TOKENS_LABEL = 'code_tokens'
RAW_TREE_LABEL = 'raw_tree'
GRAPH_LABEL = 'graph'

TreeNode = collections.OrderedDict


def _get_child_with_type(node: TreeNode, type_str: str) -> Optional[int]:
    for index, child in enumerate(node['children']):
        if child['type'] == type_str:
            return index
    return None


def _remove_docstring_node(root: TreeNode) -> Optional[str]:
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
        return expression_statement_node['children'][string_index]['string']
    except AssertionError:
        return None


def _original_to_augmented_data_path(file_path: RichPath, language: str, raw_tree_suffix: str):
    raw_tree_path = file_path.__str__().replace(f'/{language}/', f'/{language}{raw_tree_suffix}/')
    return RichPath.create(raw_tree_path)


def normalize_docstring(docstring: str):
    return ''.join(c for c in docstring if c.isalnum())


def _drop_docstring_from_graph(graph: Dict[str, Any], docstring: str):
    if 'nodes' in graph:
        docstring_index = None
        for i, node in enumerate(graph['nodes']):
            if normalize_docstring(node) == docstring:
                docstring_index = i
                break
        if docstring_index is not None:
            graph['nodes'].pop(docstring_index)

            def fix_index(v_ind):
                if int(v_ind) > docstring_index:
                    return int(v_ind) - 1
                return int(v_ind)

            if 'edges' in graph:
                for edge_type in graph['edges']:
                    graph['edges'][edge_type] = {
                        fix_index(v): [fix_index(u) for u in us if int(u) != docstring_index]
                        for v, us in graph['edges'][edge_type].items()
                        if int(v) != docstring_index
                    }


DUMMY_GRAPH = {
    'nodes': ['DUMMY', 'DUMMY'],
    'edges': {
        'CHILD': [(0, 1)]
    }
}


def _extract_graph_data(graph: Dict[str, Any]) -> Dict[str, Any]:
    is_empty = 'nodes' not in graph
    if is_empty:
        return DUMMY_GRAPH

    return {
        'nodes': graph['nodes'],
        'edges': {
            edge_type: np.array([
                [int(v), int(u)]
                for v, us in edges_of_type.items()
                for u in us
            ], dtype=np.int)
            for edge_type, edges_of_type in graph['edges'].items()
        }
    }


def combined_samples_generator(data_file: RichPath):
    raw_tree_iterator = None
    graph_iterator = None
    print(f"Generating samples from {data_file}")
    for raw_sample in data_file.read_by_file_suffix():
        assert CODE_TOKENS_LABEL in raw_sample
        if raw_tree_iterator is None:
            raw_tree_path = _original_to_augmented_data_path(
                data_file,
                language=raw_sample['language'],
                raw_tree_suffix='_raw_trees'
            )
            # raw_tree_suffix = '_compressed_100')
            raw_tree_iterator = raw_tree_path.read_by_file_suffix()
        if graph_iterator is None:
            graph_path = _original_to_augmented_data_path(
                data_file,
                language=raw_sample['language'],
                raw_tree_suffix='_graphs'
            )
            graph_iterator = graph_path.read_by_file_suffix()

        raw_tree = next(raw_tree_iterator)
        docstring = _remove_docstring_node(raw_tree)
        assert RAW_TREE_LABEL not in raw_sample
        raw_sample[RAW_TREE_LABEL] = raw_tree

        graph = next(graph_iterator)
        if docstring is not None:
            docstring = normalize_docstring(docstring)
            _drop_docstring_from_graph(graph, docstring)
        graph = _extract_graph_data(graph)

        assert GRAPH_LABEL not in raw_sample
        raw_sample[GRAPH_LABEL] = graph
        yield raw_sample
