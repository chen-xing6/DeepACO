"""
命令行评测：与 tsp/test.ipynb 流程一致，权重默认来自 ../pretrained/tsp_MCTS_teacher/。
"""
import argparse
import os
import time

import torch

from aco import ACO

EPS = 1e-10


@torch.no_grad()
def infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, k_sparse, device):
    if model is not None:
        model.eval()
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
        aco = ACO(
            n_ants=n_ants,
            heuristic=heu_mat,
            distances=distances,
            device=device,
            mcts_k_sparse=k_sparse,
        )
    else:
        aco = ACO(
            n_ants=n_ants,
            distances=distances,
            device=device,
            mcts_k_sparse=k_sparse,
        )
        if k_sparse:
            aco.sparsify(k_sparse)

    results = torch.zeros(size=(len(t_aco_diff),), device=device)
    for i, t in enumerate(t_aco_diff):
        results[i] = aco.run(t)
    return results


@torch.no_grad()
def run_test(dataset, model, n_ants, t_aco, k_sparse, device):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i + 1] - _t_aco[i] for i in range(len(_t_aco) - 1)]
    sum_results = torch.zeros(size=(len(t_aco_diff),), device=device)
    start = time.time()
    for pyg_data, distances in dataset:
        sum_results += infer_instance(
            model, pyg_data, distances, n_ants, t_aco_diff, k_sparse, device
        )
    return sum_results / len(dataset), time.time() - start


def random_smoke(device):
    torch.manual_seed(0)
    n = 12
    coords = torch.rand(n, 2, device=device)
    d = torch.norm(coords[:, None] - coords, dim=2, p=2)
    d[torch.arange(n, device=device), torch.arange(n, device=device)] = 1e10
    aco = ACO(d, n_simulations=24, mcts_k_sparse=4, device=device)
    aco.sparsify(4)
    best = aco.run(5)
    assert best < float("inf")
    print("random smoke best length:", best)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_node", type=int, default=20)
    parser.add_argument("--k_sparse", type=int, default=None)
    parser.add_argument("--n_ants", type=int, default=20)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument(
        "--t_aco",
        type=int,
        nargs="+",
        default=[1, 5, 10],
    )
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    k_sparse = args.k_sparse or max(1, args.n_node // 10)

    random_smoke(device)

    try:
        from utils import load_test_dataset
    except ModuleNotFoundError as exc:
        print("skip dataset test (utils):", exc)
        return

    data_path = os.path.join(
        os.path.dirname(__file__), f"../data/tsp/testDataset-{args.n_node}.pt"
    )
    if not os.path.isfile(data_path):
        print("skip dataset test, missing:", data_path)
        return

    test_list = load_test_dataset(args.n_node, k_sparse, device)
    model = None
    if args.model and os.path.isfile(args.model):
        from net import Net

        model = Net().to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))

    avg, duration = run_test(test_list, model, args.n_ants, args.t_aco, k_sparse, device)
    print("n_node", args.n_node, "duration", duration)
    for i, t in enumerate(args.t_aco):
        print(f"T={t}, avg cost {avg[i].item():.4f}")


if __name__ == "__main__":
    main()
