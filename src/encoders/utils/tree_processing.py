# Author: Mikita Sazanovich

import collections
from typing import Tuple, List


TreeNode = collections.OrderedDict


def try_to_queue_node(
        node: TreeNode,
        queue: collections.deque,
        nodes_queued: int,
        max_nodes: int) -> bool:
    if max_nodes == -1 or nodes_queued < max_nodes:
        queue.append(node)
        return True
    else:
        return False


def linearize_tree_bfs(
        root: TreeNode,
        max_nodes: int = -1,
        max_children: int = -1) -> Tuple[List[TreeNode], List[List[int]]]:
    nodes: List[TreeNode] = []
    children: List[List[int]] = []
    node_queue = collections.deque()
    nodes_queued = 0
    nodes_queued += try_to_queue_node(root, node_queue, nodes_queued, max_nodes)
    while node_queue:
        node = node_queue.popleft()
        node_children: List[int] = []
        node_raw_children = node['children'] if max_children == -1 else node['children'][:max_children]
        for child in node_raw_children:
            if try_to_queue_node(child, node_queue, nodes_queued, max_nodes):
                node_children.append(nodes_queued)
                nodes_queued += 1
        nodes.append(node)
        children.append(node_children)
    return nodes, children


def linearize_tree_dfs(node: TreeNode, linearization: List[TreeNode]):
    linearization.append(node)
    for child in node['children']:
        linearize_tree_dfs(child, linearization)


def get_code_tokens_from_tree(tree: TreeNode) -> List[str]:
    linearization = []
    linearize_tree_dfs(tree, linearization)
    code_tokens = []
    for node in linearization:
        node_tokens = node['string'].split()
        code_tokens.extend(node_token for node_token in node_tokens if node_token != '|')
    return code_tokens


def get_type_tokens_from_tree(tree: TreeNode) -> List[str]:
    linearization = []
    linearize_tree_dfs(tree, linearization)
    type_tokens = list(map(lambda node: node['type'], linearization))
    return type_tokens


def get_type_bag_from_tree(tree: TreeNode) -> List[str]:
    type_tokens = get_type_tokens_from_tree(tree)
    type_bag = set(type_tokens)
    return list(type_bag)
