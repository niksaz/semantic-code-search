import collections
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from dpu_utils.utils import RichPath

CODE_TOKENS_LABEL = 'code_tokens'
TREE_LABEL = '_raw_trees'
# COMPRESSED_TREE_LABEL = '_compressed_100'
GRAPH_LABEL = '_graphs'

TreeNode = collections.OrderedDict


def _get_child_with_type(graph: Dict[str, Any], node: int, type_str: str) -> Optional[int]:
    for child in graph['edges']['CHILD'].get(node, []):
        if graph['nodes'][child] == type_str:
            return child
    return None


def _get_child_with_nested_type(graph: Dict[str, Any], node: int, type_str: str, nested_type_str: str) -> Optional[int]:
    for child in graph['edges']['CHILD'].get(node, []):
        if graph['nodes'][child] == type_str:
            nested_children = graph['edges']['CHILD'].get(child, [])
            if len(nested_children) == 1 and graph['nodes'][nested_children[0]] == nested_type_str:
                return child
    return None


def _remove_docstring_from_graph(graph: Dict[str, Any]) -> Tuple[Optional[str], List[int]]:
    """The docstring node in ast3 graphs follows the structure of:
    Module
        FunctionDef
            ...
            Expr
                Str
                    LEAF
    Returns docstring and nodes to remove
    """
    try:
        root = 0
        function_definition = _get_child_with_type(graph, root, 'FunctionDef')
        if function_definition is None:
            function_definition = _get_child_with_type(graph, root, 'AsyncFunctionDef')
        assert function_definition is not None

        expr = _get_child_with_nested_type(graph, function_definition, 'Expr', 'Str')
        assert expr is not None

        str_statement = _get_child_with_type(graph, expr, 'Str')
        assert str_statement is not None

        token = _get_child_with_type(graph, str_statement, 'LEAF')
        assert token is not None

        return graph['nodes'][token], [expr, str_statement, token]
    except AssertionError:
        return None, []


def graph_to_ast(g, node_ind=0):
    children = g['edges']['CHILD'].get(node_ind, [])
    token = '' if len(children) > 0 else g['nodes'][node_ind]
    node_type = 'LEAF' if len(children) == 0 else g['nodes'][node_ind]

    for c in children:
        if g['nodes'][c] == token:
            node_type = 'UPPER_LEAF'
            token = ''

    ast = {
        'string': token,
        'type': node_type,
        'children': [
            graph_to_ast(g, c)
            for c in children
        ]
    }

    return ast


def normalize_docstring(docstring: str):
    return ''.join(c for c in docstring if c.isalnum())


def remove_nodes_from_graph(graph: Dict[str, Any], nodes: List[int]):
    if 'nodes' in graph:
        graph_indices = list(range(len(graph['nodes'])))
        for node in reversed(sorted(nodes)):
            graph['nodes'].pop(node)
            graph_indices.pop(node)

        rev_indices = {v: ind for ind, v in enumerate(graph_indices)}

        if 'edges' in graph:
            for edge_type, edges in graph['edges'].items():
                for node in nodes:
                    if node in edges:
                        del edges[node]
                for ind, v in enumerate(graph_indices):
                    if v in edges and ind != v:
                        edges[ind] = edges[v]
                        del edges[v]

                for v in edges:
                    edges[v] = [rev_indices[u] for u in edges[v] if u in rev_indices]


def normalize_graph(graph: Dict[str, Any]):
    if 'edges' in graph:
        for edge_type in graph['edges']:
            graph['edges'][edge_type] = {
                int(v): list(sorted([int(u) for u in graph['edges'][edge_type][v]]))
                for v in graph['edges'][edge_type]
            }

    if 'nodes' in graph:
        nodes_to_remove = [
            ind for ind, v in enumerate(graph['nodes'])
            if v in ['<NL>', '<INDENT>', '<DEDENT>', '(', ')', ',']
        ]
        remove_nodes_from_graph(graph, nodes_to_remove)

        if 'edges' in graph:
            _, nodes_to_remove = _remove_docstring_from_graph(graph)
            if len(nodes_to_remove) > 0:
                remove_nodes_from_graph(graph, nodes_to_remove)


DUMMY_GRAPH = {
    'nodes': ['DUMMY', 'DUMMY'],
    'edges': {
        'CHILD': [(0, 1)]
    }
}

DUMMY_AST = {
    'type': 'DUMMY',
    'string': 'DUMMY',
    'children': [
        {
            'type': 'DUMMY',
            'string': 'DUMMY',
            'children': []
        }
    ]
}


def _extract_graph_data(graph: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    is_empty = 'nodes' not in graph
    if is_empty:
        return DUMMY_GRAPH, DUMMY_AST

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
           }, graph_to_ast(graph)


def combined_samples_generator(resource_mapping: Dict[str, Optional[RichPath]]):
    language = 'python'
    # If we are training, load all available resources.
    if len(resource_mapping) == 1 and list(resource_mapping.keys())[0] == CODE_TOKENS_LABEL:
        code_data_file = resource_mapping.get(CODE_TOKENS_LABEL)
        assert code_data_file
        # for resource in [TREE_LABEL, COMPRESSED_TREE_LABEL, GRAPH_LABEL]:
        for resource in [GRAPH_LABEL]:
            resource_path = str(code_data_file).replace(f'/{language}/', f'/{language}{resource}/')
            data_file = RichPath.create(resource_path)
            resource_mapping[resource] = data_file

    resource_iterator = {}
    for resource, data_file in resource_mapping.items():
        # assert resource in [CODE_TOKENS_LABEL, TREE_LABEL, COMPRESSED_TREE_LABEL, GRAPH_LABEL]
        assert resource in [CODE_TOKENS_LABEL, GRAPH_LABEL]
        resource_iterator[resource] = data_file.read_by_file_suffix()

    while True:
        sample = {}
        try:
            # for resource in [CODE_TOKENS_LABEL, TREE_LABEL, COMPRESSED_TREE_LABEL, GRAPH_LABEL]:
            for resource in [CODE_TOKENS_LABEL, GRAPH_LABEL]:
                if resource not in resource_iterator:
                    continue
                iterator = resource_iterator[resource]
                resource_item = next(iterator)
                if resource == CODE_TOKENS_LABEL:
                    assert CODE_TOKENS_LABEL in resource_item
                    sample.update(resource_item)
                # elif resource in [TREE_LABEL, COMPRESSED_TREE_LABEL]:
                #     _remove_docstring_from_ast(resource_item)
                #     sample[resource] = resource_item
                elif resource == GRAPH_LABEL:
                    normalize_graph(resource_item)
                    graph, ast = _extract_graph_data(resource_item)
                    sample[GRAPH_LABEL] = graph
                    sample[TREE_LABEL] = ast
                else:
                    raise ValueError(f'Generator for {resource} is not implemented')
        except StopIteration:
            break
        yield sample
