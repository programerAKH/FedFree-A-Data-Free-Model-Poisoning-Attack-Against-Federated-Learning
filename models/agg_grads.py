"""聚合梯度的方法"""
import copy
from collections import OrderedDict
from typing import List, Dict, Tuple

import torch
import numpy as np


def FedAvg_grads(grads):
    avg = copy.deepcopy(grads[0])
    for k in avg:
        for i in range(1, len(grads)):
            avg[k] += grads[i][k]
        avg[k] /= len(grads)
    return avg

def krum_grads(grad_list, f):
    """
    梯度版 Krum 聚合
    :param grad_list: List[OrderedDict] 每个客户端上传的梯度字典
    :param f: 最多容忍的恶意客户端数量
    :return: 聚合后的梯度字典
    """
    n = len(grad_list)
    num_selected = n - f - 2

    client_tensors = [torch.cat([g.view(-1) for g in grad.values()]) for grad in grad_list]
    distances = []

    for i in range(n):
        dist = []
        for j in range(n):
            if i != j:
                dist.append(torch.norm(client_tensors[i] - client_tensors[j], p=2).item())
        dist.sort()
        distances.append(sum(dist[:num_selected]))

    selected_index = distances.index(min(distances))
    return grad_list[selected_index]

def multi_krum_grads(grad_list, f, m):
    n = len(grad_list)
    num_selected = n - f - 2

    client_tensors = [torch.cat([g.view(-1) for g in grad.values()]) for grad in grad_list]
    distances = []

    for i in range(n):
        dist = []
        for j in range(n):
            if i != j:
                dist.append(torch.norm(client_tensors[i] - client_tensors[j], p=2).item())
        dist.sort()
        distances.append(sum(dist[:num_selected]))

    selected_indices = sorted(range(n), key=lambda i: distances[i])[:m]
    selected_grads = [grad_list[i] for i in selected_indices]

    # 聚合被选中的梯度字典
    aggregated = OrderedDict()
    for key in selected_grads[0]:
        aggregated[key] = torch.mean(torch.stack([g[key] for g in selected_grads]), dim=0)
    return aggregated

def trimmed_mean_grads(grad_list, beta):
    """
    :param grad_list: List[OrderedDict]
    :param beta: 去掉最大最小的数量
    :return: 聚合梯度
    """
    aggregated = OrderedDict()
    for key in grad_list[0]:
        stacked = torch.stack([grad[key] for grad in grad_list])
        sorted_vals, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_vals[beta:-beta]
        aggregated[key] = torch.mean(trimmed, dim=0)
    return aggregated

def bulyan_grads(grad_list, f, m, beta):
    n = len(grad_list)
    num_selected = n - f - 2

    client_tensors = [torch.cat([g.view(-1) for g in grad.values()]) for grad in grad_list]
    distances = []

    for i in range(n):
        dist = []
        for j in range(n):
            if i != j:
                dist.append(torch.norm(client_tensors[i] - client_tensors[j], p=2).item())
        dist.sort()
        distances.append(sum(dist[:num_selected]))

    selected_indices = sorted(range(n), key=lambda i: distances[i])[:m]
    selected_grads = [grad_list[i] for i in selected_indices]

    return trimmed_mean_grads(selected_grads, beta)

class RFA_Grad:
    def __init__(self, max_iter: int = 10, eps: float = 1e-6, nu: float = 1e-6):
        self.max_iter = max_iter
        self.eps = eps
        self.nu = nu

    def _flatten_grads(self, grad_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([g.flatten() for g in grad_dict.values()])

    def _unflatten_grads(self, flat_tensor: torch.Tensor, template: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        new_grads = {}
        idx = 0
        for k, v in template.items():
            numel = v.numel()
            new_grads[k] = flat_tensor[idx:idx + numel].view_as(v).clone()
            idx += numel
        return new_grads

    def aggregate(self, grad_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        n = len(grad_list)
        device = next(iter(grad_list[0].values())).device
        weights = torch.ones(n, device=device) / n

        flat_grads = [self._flatten_grads(g).to(device) for g in grad_list]
        stacked = torch.stack(flat_grads)
        v = torch.mean(stacked, dim=0)  # 初始聚合点

        for _ in range(self.max_iter):
            distances = torch.norm(stacked - v, dim=1).clamp(min=self.nu)
            beta = weights / distances
            beta_sum = beta.sum()
            new_v = torch.sum((beta[:, None] * stacked), dim=0) / beta_sum
            if torch.norm(new_v - v) < self.eps:
                break
            v = new_v

        return self._unflatten_grads(v, grad_list[0])

class CenteredClipping_Grad:
    def __init__(self, tau: float = 100.0, num_iter: int = 1):
        self.tau = tau
        self.num_iter = num_iter

    def _flatten_grads(self, grad_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([v.view(-1) for v in grad_dict.values()])

    def _unflatten_grads(self, flat_tensor: torch.Tensor, template: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        grad_dict = {}
        pointer = 0
        for k, v in template.items():
            numel = v.numel()
            grad_dict[k] = flat_tensor[pointer:pointer + numel].view_as(v)
            pointer += numel
        return grad_dict

    def aggregate(self, grad_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        deltas = [self._flatten_grads(grad) for grad in grad_list]
        v = torch.zeros_like(deltas[0])

        for _ in range(self.num_iter):
            updates = []
            for d in deltas:
                diff = d - v
                norm = torch.norm(diff) + 1e-6
                clipped = diff * min(1.0, self.tau / norm)
                updates.append(clipped)
            v = v + sum(updates) / len(updates)

        return self._unflatten_grads(v, grad_list[0])

class RoFL_Grad:
    def __init__(self, norm_type="L2", r=2.0, dynamic_bound=True, ema_beta=0.8):
        self.norm_type = norm_type
        self.r = r
        self.dynamic_bound = dynamic_bound
        self.ema_beta = ema_beta
        self.smoothed_bound = None

    def _flatten(self, grad_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([v.detach().cpu().view(-1) for v in grad_dict.values()])

    def _unflatten(self, flat: torch.Tensor, template: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        new_grad = {}
        ptr = 0
        for k, v in template.items():
            numel = v.numel()
            new_grad[k] = flat[ptr:ptr + numel].view_as(v)
            ptr += numel
        return new_grad

    def _compute_norm(self, vec: np.ndarray) -> float:
        if self.norm_type == "L2":
            return np.linalg.norm(vec)
        elif self.norm_type == "Linf":
            return np.max(np.abs(vec))
        else:
            raise ValueError("Unsupported norm type")

    def _clip_soft(self, vec: np.ndarray, bound: float) -> np.ndarray:
        norm = self._compute_norm(vec)
        if norm <= bound:
            return vec
        scale = np.tanh(bound / norm)
        return vec * scale

    def _robust_median(self, norms: List[float]) -> float:
        norms = np.array(norms)
        lower, upper = np.percentile(norms, [10, 90])
        filtered = norms[(norms >= lower) & (norms <= upper)]
        return np.median(filtered)

    def aggregate(self, grads: List[Dict[str, torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], float]:
        flat_grads = [self._flatten(g).numpy() for g in grads]
        norms = [self._compute_norm(g) for g in flat_grads]

        # 动态边界估计
        if self.dynamic_bound:
            median = self._robust_median(norms)
            bound = self.r * median
            if self.smoothed_bound is None:
                self.smoothed_bound = bound
            else:
                bound = self.ema_beta * self.smoothed_bound + (1 - self.ema_beta) * bound
                self.smoothed_bound = bound
        else:
            bound = self.smoothed_bound or np.median(norms)

        # 对每个梯度进行 soft clipping
        clipped = [self._clip_soft(g, bound) for g in flat_grads]
        avg_grad_np = np.mean(clipped, axis=0)
        avg_grad_tensor = torch.tensor(avg_grad_np)

        return self._unflatten(avg_grad_tensor, grads[0]), bound


