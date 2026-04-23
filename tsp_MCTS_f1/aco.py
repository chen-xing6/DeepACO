import torch
from torch.distributions import Categorical
import math
import random

class MCTSNode:
    def __init__(self, state, mcts, parent=None, action=None):
        self.state = state  # partial tour as list of indices
        self.mcts = mcts
        self.parent = parent
        self.action = action  # the action that led to this node
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = self.mcts.get_legal_actions(state) if mcts else None  # will be set when expanded

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c=1.4):
        if not self.children:
            return None
        choices_weights = [
            (child.value / child.visits) + c * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self, action, state):
        child = MCTSNode(state, self.mcts, parent=self, action=action)
        self.children.append(child)
        return child

class MCTS:
    def __init__(self, distances, n_simulations=100, pheromone=None, heuristic=None, alpha=1, beta=1):
        self.distances = distances
        self.n_simulations = n_simulations
        self.n_nodes = len(distances)
        self.pheromone = pheromone
        self.heuristic = heuristic
        self.alpha = alpha
        self.beta = beta
        # 可选的轻量缓存：tuple(state)->best_action
        # 注意：当 pheromone/heuristic/alpha/beta 变化时，应使用新的 cache 或清空 cache
        self._cache = None
        # 可选：tuple(state)->teacher 分布向量 (n_nodes,) 供 KL 蒸馏缓存
        self._soft_cache = None

    def _prior_pick(self, state, legal_actions):
        """用 pheromone×heuristic 作为先验，在扩展时优先选择更可能的动作。"""
        if not legal_actions:
            return None
        if self.pheromone is None or self.heuristic is None:
            return random.choice(legal_actions)
        current = state[-1]
        legal = torch.tensor(legal_actions, device=self.distances.device, dtype=torch.long)
        w = (self.pheromone[current, legal] ** self.alpha) * (self.heuristic[current, legal] ** self.beta)
        w = w.clamp(min=1e-9)
        probs = w / w.sum()
        idx = torch.multinomial(probs, num_samples=1).item()
        return legal_actions[idx]

    def get_legal_actions(self, state):
        visited = set(state)
        return [i for i in range(self.n_nodes) if i not in visited]

    def simulate(self, state):
        if len(state) == self.n_nodes:
            # Complete tour, calculate cost
            cost = 0.0
            for i in range(self.n_nodes):
                cost += self.distances[state[i], state[(i + 1) % self.n_nodes]].item()
            return -cost

        current = state[-1]
        visited = set(state)
        path = state[:]
        cost = 0.0

        while len(path) < self.n_nodes:
            legal = [i for i in range(self.n_nodes) if i not in visited]

            if self.pheromone is not None and self.heuristic is not None:
                w = (self.pheromone[current, legal] ** self.alpha) * (self.heuristic[current, legal] ** self.beta)
                w = w.clamp(min=1e-9)
                probs = w / w.sum()
                idx = torch.multinomial(probs, num_samples=1).item()
                next_city = legal[idx]
            else:
                # fallback: distance-based greedy soft selection
                dists = self.distances[current, legal]
                inv = 1.0 / (dists + 1e-9)
                probs = inv / inv.sum()
                idx = torch.multinomial(probs, num_samples=1).item()
                next_city = legal[idx]

            cost += self.distances[current, next_city].item()
            path.append(next_city)
            visited.add(next_city)
            current = next_city

        cost += self.distances[current, state[0]].item()
        return -cost  # negative cost as reward (maximize)

    def _simulate_tree(self, root_state):
        """运行完整 MCTS 模拟，返回根节点（含根下子节点 visit 统计）。"""
        root = MCTSNode(root_state, self)
        for _ in range(self.n_simulations):
            node = root
            state = root_state[:]
            while node.is_fully_expanded() and node.children and len(state) < self.n_nodes:
                node = node.best_child()
                state = node.state[:]
            if len(state) == self.n_nodes:
                reward = self.simulate(state)
                back_node = node
            else:
                if not node.is_fully_expanded():
                    # 用先验引导扩展，低模拟次数下更稳
                    action = self._prior_pick(state, node.untried_actions)
                    node.untried_actions.remove(action)
                    new_state = state + [action]
                    node = node.expand(action, new_state)
                    state = new_state
                reward = self.simulate(state)
                back_node = node
            while back_node:
                back_node.visits += 1
                back_node.value += reward
                back_node = back_node.parent
        return root

    def search(self, root_state):
        if self._cache is not None:
            key = tuple(root_state)
            cached = self._cache.get(key, None)
            if cached is not None:
                return cached
        root = self._simulate_tree(root_state)
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            best_action = best_child.action
        else:
            best_action = random.choice(root.untried_actions)

        if self._cache is not None:
            self._cache[tuple(root_state)] = best_action
        return best_action

    def root_visit_policy_tensor(
        self,
        root_state,
        device,
        dtype=torch.float32,
        visit_eps=1e-3,
    ):
        """
        用根节点各子节点的访问次数构造老师分布（在合法动作上归一化）。
        visit_eps 给未展开/零 visit 的合法边一个下限，避免 one-hot 退化为纯 hard label。
        """
        key = tuple(root_state)
        if self._soft_cache is not None and key in self._soft_cache:
            return self._soft_cache[key].to(device=device, dtype=dtype)

        root = self._simulate_tree(root_state)
        legal = self.get_legal_actions(root_state)
        n = self.n_nodes
        probs = torch.zeros(n, device=device, dtype=dtype)
        for a in legal:
            probs[a] = visit_eps
        if root.children:
            for c in root.children:
                probs[c.action] = probs[c.action] + float(c.visits)
        else:
            probs.zero_()
            for a in legal:
                probs[a] = 1.0
        probs = probs / probs.sum().clamp(min=1e-12)

        if self._soft_cache is not None:
            self._soft_cache[key] = probs.detach().cpu()
        return probs

    def set_cache(self, cache: dict | None):
        """设置/清空搜索缓存。cache 应为 dict: tuple(state)->action。"""
        self._cache = cache

    def set_soft_cache(self, cache: dict | None):
        """设置/清空软标签缓存。cache 应为 dict: tuple(state)->CPU tensor (n_nodes,)。"""
        self._soft_cache = cache

class ACO():

    def __init__(self, 
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
                 device='cpu',
                 use_mcts=False,
                 mcts_simulations=100
                 ):
        
        self.problem_size = len(distances)
        self.distances  = distances
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist
        self.min_max = min_max
        
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
            self.pheromone = pheromone

        self.heuristic = 1 / distances if heuristic is None else heuristic

        self.shortest_path = None
        self.lowest_cost = float('inf')

        self.device = device

        self.use_mcts = use_mcts
        if self.use_mcts:
            self.mcts = MCTS(
                distances,
                mcts_simulations,
                pheromone=self.pheromone,
                heuristic=self.heuristic,
                alpha=self.alpha,
                beta=self.beta,
            )

    @torch.no_grad()
    def sparsify(self, k_sparse):
        '''
        Sparsify the TSP graph to obtain the heuristic information 
        Used for vanilla ACO baselines
        '''
        _, topk_indices = torch.topk(self.distances, 
                                        k=k_sparse, 
                                        dim=1, largest=False)
        edge_index_u = torch.repeat_interleave(
            torch.arange(len(self.distances), device=self.device),
            repeats=k_sparse
            )
        edge_index_v = torch.flatten(topk_indices)
        sparse_distances = torch.ones_like(self.distances) * 1e10
        sparse_distances[edge_index_u, edge_index_v] = self.distances[edge_index_u, edge_index_v]
        self.heuristic = 1 / sparse_distances
    
    def sample(self):
        paths, log_probs = self.gen_path(require_prob=True)
        costs = self.gen_path_costs(paths)
        return costs, log_probs

    @torch.no_grad()
    def run(self, n_iterations):
        for _ in range(n_iterations):
            paths = self.gen_path(require_prob=False)
            costs = self.gen_path_costs(paths)
            
            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost
                if self.min_max:
                    max = self.problem_size / self.lowest_cost
                    if self.max is None:
                        self.pheromone *= max/self.pheromone.max()
                    self.max = max
            
            self.update_pheronome(paths, costs)

        return self.lowest_cost
       
    @torch.no_grad()
    def update_pheronome(self, paths, costs):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        self.pheromone = self.pheromone * self.decay 
        
        if self.elitist:
            best_cost, best_idx = costs.min(dim=0)
            best_tour= paths[:, best_idx]
            self.pheromone[best_tour, torch.roll(best_tour, shifts=1)] += 1.0/best_cost
            self.pheromone[torch.roll(best_tour, shifts=1), best_tour] += 1.0/best_cost
        
        else:
            for i in range(self.n_ants):
                path = paths[:, i]
                cost = costs[i]
                self.pheromone[path, torch.roll(path, shifts=1)] += 1.0/cost
                self.pheromone[torch.roll(path, shifts=1), path] += 1.0/cost
                
        if self.min_max:
            self.pheromone[(self.pheromone > 1e-9) * (self.pheromone) < self.min] = self.min
            self.pheromone[self.pheromone > self.max] = self.max
    
    @torch.no_grad()
    def gen_path_costs(self, paths):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
        Returns:
                Lengths of paths: torch tensor with shape (n_ants,)
        '''
        assert paths.shape == (self.problem_size, self.n_ants)
        u = paths.T # shape: (n_ants, problem_size)
        v = torch.roll(u, shifts=1, dims=1)  # shape: (n_ants, problem_size)
        assert (self.distances[u, v] > 0).all()
        return torch.sum(self.distances[u, v], dim=1)

    def gen_path(self, require_prob=False):
        '''
        Tour contruction for all ants
        Returns:
            paths: torch tensor with shape (problem_size, n_ants), paths[:, i] is the constructed tour of the ith ant
            log_probs: torch tensor with shape (problem_size, n_ants), log_probs[i, j] is the log_prob of the ith action of the jth ant
        '''
        start = torch.randint(low=0, high=self.problem_size, size=(self.n_ants,), device=self.device)
        mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        mask[torch.arange(self.n_ants, device=self.device), start] = 0
        
        paths_list = [] # paths_list[i] is the ith move (tensor) for all ants
        paths_list.append(start)
        
        log_probs_list = [] # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions
        
        current_paths = [[start[i].item()] for i in range(self.n_ants)]  # list of lists for current paths
        
        prev = start
        for _ in range(self.problem_size-1):
            actions, log_probs = self.pick_move(prev, mask, require_prob, current_paths)
            paths_list.append(actions)
            if require_prob:
                if log_probs is None:
                    log_probs = torch.zeros((self.n_ants,), device=self.device)
                log_probs_list.append(log_probs)
                mask = mask.clone()
            prev = actions
            mask[torch.arange(self.n_ants, device=self.device), actions] = 0
            # Update current_paths
            for i in range(self.n_ants):
                current_paths[i].append(actions[i].item())
        
        if require_prob:
            return torch.stack(paths_list), torch.stack(log_probs_list)
        else:
            return torch.stack(paths_list)
        
    def pick_move(self, prev, mask, require_prob, current_paths=None):
        '''
        Args:
            prev: tensor with shape (n_ants,), previous nodes for all ants
            mask: bool tensor with shape (n_ants, p_size), masks (0) for the visited cities
            current_paths: list of lists, current paths for each ant
        '''
        if self.use_mcts and current_paths is not None:
            # Keep MCTS heuristics in sync with current ACO pheromones/heuristic
            self.mcts.pheromone = self.pheromone
            self.mcts.heuristic = self.heuristic
            self.mcts.alpha = self.alpha
            self.mcts.beta = self.beta
            actions = []
            log_probs = [] if require_prob else None
            for i in range(self.n_ants):
                action = self.mcts.search(current_paths[i])
                actions.append(action)

                if require_prob:
                    # estimate policy log_prob under current distribution (for policy gradient)
                    pheromone_i = self.pheromone[prev[i]]
                    heuristic_i = self.heuristic[prev[i]]
                    dist_i = ((pheromone_i ** self.alpha) * (heuristic_i ** self.beta) * mask[i])
                    dist_i = Categorical(dist_i)
                    log_probs.append(dist_i.log_prob(torch.tensor(action, device=self.device)))

            actions = torch.tensor(actions, device=self.device)
            if require_prob:
                log_probs = torch.stack(log_probs)
        else:
            pheromone = self.pheromone[prev] # shape: (n_ants, p_size)
            heuristic = self.heuristic[prev] # shape: (n_ants, p_size)
            dist = ((pheromone ** self.alpha) * (heuristic ** self.beta) * mask) # shape: (n_ants, p_size)
            dist = Categorical(dist)
            actions = dist.sample() # shape: (n_ants,)
            log_probs = dist.log_prob(actions) if require_prob else None # shape: (n_ants,)
        return actions, log_probs
        


if __name__ == '__main__':
    torch.set_printoptions(precision=3,sci_mode=False)
    input = torch.rand(size=(5, 2))
    distances = torch.norm(input[:, None] - input, dim=2, p=2)
    distances[torch.arange(len(distances)), torch.arange(len(distances))] = 1e10
    aco = ACO(distances, use_mcts=True, mcts_simulations=10)
    aco.sparsify(k_sparse=3)
    print(aco.run(20))