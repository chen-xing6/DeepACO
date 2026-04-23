"""
TSP 上的 MCTS：状态为 (current, visited_mask, start)，PUCT 先验由信息素 × 启发式给出。
n 可用 Python int 位掩码表示已访问集合（任意规模均可）；默认建议 n 较大时配合 k_sparse 降分支。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TSPMCTSContext:
    n: int
    distances: np.ndarray
    heuristic: np.ndarray
    tau: np.ndarray
    alpha: float
    beta: float
    c_puct: float
    neighbors: List[List[int]]
    anchor_i: int
    anchor_j: int
    anchor_boost: float
    eps: float = 1e-10

    def all_visited_mask(self) -> int:
        return (1 << self.n) - 1

    def legal_moves(self, current: int, visited: int) -> List[int]:
        full_mask = self.all_visited_mask()
        unvisited = [i for i in range(self.n) if not (visited >> i) & 1]
        if visited == full_mask:
            return []
        cand = [i for i in self.neighbors[current] if not (visited >> i) & 1]
        if not cand:
            cand = unvisited
        return cand

    def prior_for_moves(
        self, current: int, visited: int, moves: List[int]
    ) -> np.ndarray:
        if not moves:
            return np.array([], dtype=np.float64)
        t_row = np.maximum(self.tau[current, moves], self.eps)
        h_row = np.maximum(self.heuristic[current, moves], self.eps)
        p = (t_row ** self.alpha) * (h_row ** self.beta)
        if current == self.anchor_i and self.anchor_j in moves:
            idx = moves.index(self.anchor_j)
            p[idx] *= self.anchor_boost
        s = p.sum()
        if s < self.eps:
            p = np.ones(len(moves), dtype=np.float64) / len(moves)
        else:
            p = p / s
        return p


class MCTSNode:
    __slots__ = (
        "current",
        "visited",
        "start",
        "parent",
        "move_from_parent",
        "children",
        "untried",
        "N",
        "W",
    )

    def __init__(
        self,
        current: int,
        visited: int,
        start: int,
        parent: Optional[MCTSNode],
        move_from_parent: Optional[int],
        untried: List[int],
    ):
        self.current = current
        self.visited = visited
        self.start = start
        self.parent = parent
        self.move_from_parent = move_from_parent
        self.children: Dict[int, MCTSNode] = {}
        self.untried = untried
        self.N = 0
        self.W = 0.0

    def is_terminal(self, ctx: TSPMCTSContext) -> bool:
        return self.visited == ctx.all_visited_mask()

    def expand(self, ctx: TSPMCTSContext) -> MCTSNode:
        a = self.untried.pop()
        visited_new = self.visited | (1 << a)
        child_untried = list(ctx.legal_moves(a, visited_new))
        random.shuffle(child_untried)
        child = MCTSNode(
            current=a,
            visited=visited_new,
            start=self.start,
            parent=self,
            move_from_parent=a,
            untried=child_untried,
        )
        self.children[a] = child
        return child

    def best_child(self, ctx: TSPMCTSContext) -> Tuple[int, MCTSNode]:
        moves = list(self.children.keys())
        priors = ctx.prior_for_moves(self.current, self.visited, moves)
        total_n = max(self.N, 1)
        sqrt_n = math.sqrt(total_n)
        best_a = moves[0]
        best_score = -float("inf")
        for i, a in enumerate(moves):
            ch = self.children[a]
            q = ch.W / ch.N if ch.N > 0 else 0.0
            u = ctx.c_puct * float(priors[i]) * sqrt_n / (1 + ch.N)
            score = q + u
            if score > best_score:
                best_score = score
                best_a = a
        return best_a, self.children[best_a]


def _rollout(
    ctx: TSPMCTSContext,
    current: int,
    visited: int,
    start: int,
    partial_cost: float,
) -> Tuple[float, List[int]]:
    """从 (current, visited) 随机策略补全回路，返回 (总代价, 完整城市序列，从 start 起)."""
    path = []
    cur = current
    vis = visited
    cost = partial_cost
    full = ctx.all_visited_mask()

    while vis != full:
        moves = ctx.legal_moves(cur, vis)
        if not moves:
            moves = [i for i in range(ctx.n) if not (vis >> i) & 1]
        pri = ctx.prior_for_moves(cur, vis, moves)
        a = int(np.random.choice(moves, p=pri))
        cost += float(ctx.distances[cur, a])
        path.append(a)
        cur = a
        vis |= 1 << a

    cost += float(ctx.distances[cur, start])
    full_path = [start]
    # 从 root 到 current 的路径需由调用方前置拼接；此处 path 仅为 rollout 段
    return cost, path


def backprop(path_nodes: List[MCTSNode], reward: float) -> None:
    for nd in path_nodes:
        nd.N += 1
        nd.W += reward


def run_simulations(
    ctx: TSPMCTSContext,
    n_simulations: int,
) -> Tuple[float, List[int]]:
    """
    从锚定根状态 (anchor_i 已访问，当前在 anchor_i) 执行 n_simulations 次 MCTS。
    返回全局本轮最优 (长度, 路径序列，以 start 为首、长度为 n).
    """
    root_moves = ctx.legal_moves(ctx.anchor_i, 1 << ctx.anchor_i)
    if not root_moves:
        return float("inf"), []
    untried = [m for m in root_moves]
    random.shuffle(untried)
    root = MCTSNode(
        current=ctx.anchor_i,
        visited=1 << ctx.anchor_i,
        start=ctx.anchor_i,
        parent=None,
        move_from_parent=None,
        untried=untried,
    )

    best_L = float("inf")
    best_path: List[int] = []

    for _ in range(n_simulations):
        node = root
        cost = 0.0
        path_nodes: List[MCTSNode] = [root]

        while not node.is_terminal(ctx):
            if node.untried:
                child = node.expand(ctx)
                cost += float(ctx.distances[node.current, child.current])
                path_nodes.append(child)
                node = child
                break
            if not node.children:
                break
            a, node = node.best_child(ctx)
            cost += float(ctx.distances[path_nodes[-1].current, node.current])
            path_nodes.append(node)

        if node.is_terminal(ctx):
            total_L = cost + float(ctx.distances[node.current, node.start])
            reward = -total_L
            backprop(path_nodes, reward)
            if total_L < best_L:
                best_L = total_L
                best_path = [n.current for n in path_nodes]
            continue

        total_L, roll_rest = _rollout(
            ctx, node.current, node.visited, node.start, cost
        )
        reward = -total_L
        seq_nodes = path_nodes
        backprop(seq_nodes, reward)
        if total_L < best_L:
            best_L = total_L
            best_path = [n.current for n in seq_nodes] + roll_rest

    return best_L, best_path


def build_neighbors_from_distances(
    distances: np.ndarray, k_sparse: Optional[int]
) -> List[List[int]]:
    n = distances.shape[0]
    if k_sparse is None or k_sparse >= n:
        return [list(j for j in range(n) if j != i) for i in range(n)]
    neigh: List[List[int]] = []
    for i in range(n):
        row = distances[i].copy()
        row[i] = np.inf
        idx = np.argpartition(row, min(k_sparse, n - 1) - 1)[:k_sparse]
        idx = idx[np.argsort(row[idx])]
        neigh.append([int(j) for j in idx if j != i])
    return neigh


def tour_length(distances: np.ndarray, tour: List[int]) -> float:
    if len(tour) != len(set(tour)) or len(tour) != distances.shape[0]:
        return float("inf")
    s = 0.0
    for i in range(len(tour)):
        a = tour[i]
        b = tour[(i + 1) % len(tour)]
        s += float(distances[a, b])
    return s
