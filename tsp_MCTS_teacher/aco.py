"""
DeepACO 变体：训练阶段与原版相同（多蚁按 τ×η 策略采样 + REINFORCE）；
run() 阶段用「全局信息素 τ + 锚定边 PUCT-MCTS」搜索并反哺 τ。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from mcts_tsp import (
    TSPMCTSContext,
    build_neighbors_from_distances,
    run_simulations,
    tour_length,
)


class ACO:
    def __init__(
        self,
        distances,
        n_ants=20,
        decay=0.9,
        alpha=1,
        beta=1,
        elitist=False,
        min_max=False,
        pheromone=None,
        heuristic=None,
        min=None,
        device="cpu",
        n_simulations=64,
        c_puct=1.5,
        anchor_boost=2.0,
        mcts_k_sparse=None,
        eps=1e-10,
    ):
        self.problem_size = len(distances)
        self.distances = distances.to(device)
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist
        self.min_max = min_max
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.anchor_boost = anchor_boost
        self.mcts_k_sparse = mcts_k_sparse
        self.eps = eps

        if min_max:
            if min is not None:
                assert min > 1e-9
            else:
                min = 0.1
            self.min = min
            self.max = None

        if pheromone is None:
            self.pheromone = torch.ones_like(self.distances)
            if min_max:
                self.pheromone = self.pheromone * self.min
        else:
            self.pheromone = pheromone.to(device)

        self.heuristic = (
            1 / self.distances if heuristic is None else heuristic.to(device)
        )

        self.shortest_path = None
        self.lowest_cost = float("inf")

        self.device = device
        self._dist_np = self.distances.detach().cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def sparsify(self, k_sparse):
        _, topk_indices = torch.topk(
            self.distances, k=k_sparse, dim=1, largest=False
        )
        edge_index_u = torch.repeat_interleave(
            torch.arange(len(self.distances), device=self.device),
            repeats=k_sparse,
        )
        edge_index_v = torch.flatten(topk_indices)
        sparse_distances = torch.ones_like(self.distances) * 1e10
        sparse_distances[edge_index_u, edge_index_v] = self.distances[
            edge_index_u, edge_index_v
        ]
        self.heuristic = 1 / sparse_distances

    def sample(self):
        paths, log_probs = self.gen_path(require_prob=True)
        costs = self.gen_path_costs(paths)
        return costs, log_probs

    def _sample_anchor_edge(self) -> Tuple[int, int]:
        tau = self.pheromone.detach().cpu().numpy().astype(np.float64)
        eta = self.heuristic.detach().cpu().numpy().astype(np.float64)
        w = (tau ** self.alpha) * (eta ** self.beta)
        np.fill_diagonal(w, 0.0)
        flat = w.ravel()
        s = flat.sum()
        n = self.problem_size
        if s < self.eps:
            i = int(np.random.randint(0, n))
            choices = [j for j in range(n) if j != i]
            j = int(np.random.choice(choices))
            return i, j
        p = flat / s
        idx = int(np.random.choice(n * n, p=p))
        return idx // n, idx % n

    @torch.no_grad()
    def _mcts_deposit(self, tour: List[int], cost: float) -> None:
        if cost <= 0 or not tour or len(tour) != self.problem_size:
            return
        t = torch.tensor(tour, device=self.device, dtype=torch.long)
        delta = 1.0 / cost
        self.pheromone[t, torch.roll(t, shifts=1)] += delta
        self.pheromone[torch.roll(t, shifts=1), t] += delta

    @torch.no_grad()
    def _clip_pheromone(self) -> None:
        if not self.min_max:
            return
        self.pheromone[(self.pheromone > 1e-9) * (self.pheromone < self.min)] = (
            self.min
        )
        if self.max is not None:
            self.pheromone[self.pheromone > self.max] = self.max

    @torch.no_grad()
    def run(self, n_iterations):
        self._dist_np = self.distances.detach().cpu().numpy().astype(np.float64)
        neighbors = build_neighbors_from_distances(
            self._dist_np, self.mcts_k_sparse
        )
        n = self.problem_size

        for _ in range(n_iterations):
            self.pheromone *= self.decay

            anchor_i, anchor_j = self._sample_anchor_edge()
            if anchor_i == anchor_j:
                choices = [j for j in range(n) if j != anchor_i]
                anchor_j = int(np.random.choice(choices))

            tau_np = self.pheromone.detach().cpu().numpy().astype(np.float64)
            heu_np = self.heuristic.detach().cpu().numpy().astype(np.float64)

            ctx = TSPMCTSContext(
                n=n,
                distances=self._dist_np,
                heuristic=heu_np,
                tau=tau_np,
                alpha=self.alpha,
                beta=self.beta,
                c_puct=self.c_puct,
                neighbors=neighbors,
                anchor_i=anchor_i,
                anchor_j=anchor_j,
                anchor_boost=self.anchor_boost,
                eps=self.eps,
            )

            best_L, best_path = run_simulations(ctx, self.n_simulations)

            if best_path and len(best_path) == n:
                L_verify = tour_length(self._dist_np, best_path)
                if L_verify < float("inf"):
                    best_L = min(best_L, L_verify)
                if best_L < self.lowest_cost:
                    self.lowest_cost = torch.as_tensor(
                        best_L, device=self.device, dtype=self.distances.dtype
                    )
                    self.shortest_path = torch.tensor(
                        best_path, device=self.device, dtype=torch.long
                    )
                    if self.min_max:
                        max_v = self.problem_size / self.lowest_cost
                        if self.max is None:
                            self.pheromone *= max_v / self.pheromone.max()
                        self.max = max_v
                self._mcts_deposit(best_path, best_L)

            self._clip_pheromone()

        return self.lowest_cost

    @torch.no_grad()
    def update_pheronome(self, paths, costs):
        self.pheromone = self.pheromone * self.decay

        if self.elitist:
            best_cost, best_idx = costs.min(dim=0)
            best_tour = paths[:, best_idx]
            self.pheromone[best_tour, torch.roll(best_tour, shifts=1)] += (
                1.0 / best_cost
            )
            self.pheromone[torch.roll(best_tour, shifts=1), best_tour] += (
                1.0 / best_cost
            )
        else:
            for i in range(self.n_ants):
                path = paths[:, i]
                cost = costs[i]
                self.pheromone[path, torch.roll(path, shifts=1)] += 1.0 / cost
                self.pheromone[torch.roll(path, shifts=1), path] += 1.0 / cost

        if self.min_max:
            self.pheromone[
                (self.pheromone > 1e-9) * (self.pheromone < self.min)
            ] = self.min
            self.pheromone[self.pheromone > self.max] = self.max

    @torch.no_grad()
    def gen_path_costs(self, paths):
        assert paths.shape == (self.problem_size, self.n_ants)
        u = paths.T
        v = torch.roll(u, shifts=1, dims=1)
        assert (self.distances[u, v] > 0).all()
        return torch.sum(self.distances[u, v], dim=1)

    def gen_path(self, require_prob=False):
        start = torch.randint(
            low=0, high=self.problem_size, size=(self.n_ants,), device=self.device
        )
        mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        mask[torch.arange(self.n_ants, device=self.device), start] = 0

        paths_list = [start]
        log_probs_list = []
        prev = start
        for _ in range(self.problem_size - 1):
            actions, log_probs = self.pick_move(prev, mask, require_prob)
            paths_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)
                mask = mask.clone()
            prev = actions
            mask[torch.arange(self.n_ants, device=self.device), actions] = 0

        if require_prob:
            return torch.stack(paths_list), torch.stack(log_probs_list)
        return torch.stack(paths_list)

    def pick_move(self, prev, mask, require_prob):
        pheromone = self.pheromone[prev]
        heuristic = self.heuristic[prev]
        dist = (pheromone ** self.alpha) * (heuristic ** self.beta) * mask
        dist = Categorical(dist)
        actions = dist.sample()
        log_probs = dist.log_prob(actions) if require_prob else None
        return actions, log_probs


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False)
    inp = torch.rand(size=(8, 2))
    distances = torch.norm(inp[:, None] - inp, dim=2, p=2)
    distances[torch.arange(8), torch.arange(8)] = 1e10
    aco = ACO(distances, n_simulations=32, mcts_k_sparse=4, device="cpu")
    aco.sparsify(4)
    print("mcts run", aco.run(8))
