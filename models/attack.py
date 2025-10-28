import torch
import numpy as np
from collections import deque
from typing import Dict

from models.shampoo import data_less_local_train
from utils.parameters import args_parser

args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


def compute_delta_data_free(
    current_params: dict,
    historical_grads: Dict[str, torch.Tensor],
    states: Dict[str, Dict],
    num_samples: int = 5,
    delta_max: float = 10.0,
    tau: float = 0.01,
):
    benign_updates_list = []
    for i in range(num_samples):
        simulated_params, _ = data_less_local_train(current_params, historical_grads, states, t=i+1)
        update_vec = torch.cat([
            (simulated_params[name] - current_params[name]).flatten()
            for name in current_params
        ])
        benign_updates_list.append(update_vec)

    benign_tensor = torch.stack(benign_updates_list)
    mean_update = benign_tensor.mean(dim=0)
    std_update = benign_tensor.std(dim=0)

    def generate_malicious_update(delta_val):
        return mean_update - delta_val * std_update

    def is_aggregated(candidate):
        distances = [torch.norm(candidate - b) for b in benign_updates_list]
        threshold = torch.mean(torch.tensor(distances))
        return sum(d < threshold for d in distances) >= len(benign_updates_list) / 2

    low, high = 0.0, delta_max
    while high - low > tau:
        mid = (low + high) / 2
        candidate = generate_malicious_update(mid)
        if is_aggregated(candidate):
            low = mid
        else:
            high = mid

    return (low + high) / 2


class MaliciousParams:
    def __init__(self,safe_max_ratio,center,scale,safe_min_ratio,use_adaptive_norm):
        self.benign_model_queue = deque(maxlen=5)
        self.attack_direction_queue = deque(maxlen=5)
        self.update_stats = {}
        self.delta_history = deque(maxlen=3)
        self.round = 1

        # norm 参数
        self.initial_norm = 1.2
        self.distance_threshold = 1.2
        self.std_weight = 0.1
        self.safe_min_ratio = safe_min_ratio
        self.safe_max_ratio = safe_max_ratio#mnist：7.6，f-mnist：1.6，cifar10：1.86
        self._cached_benign_avg = None
        #cifar10：12-8，#f-mnist：12-15 ，mnist：30-20
        self.center = center
        self.scale = scale
        self.use_adaptive_norm = use_adaptive_norm


    def _compute_max_norm(self, benign_model_queue, malicious_update):
        if not benign_model_queue:
            return self.safe_min_ratio * self.initial_norm

        if self._cached_benign_avg is None:
            self._cached_benign_avg = self._average_model(list(benign_model_queue))

        # === 1. 计算 benign 模型的 norm 分布 ===
        benign_norms = [torch.norm(b[name]).item() for b in benign_model_queue for name in b]
        mean_norm = np.mean(benign_norms)
        std_norm = np.std(benign_norms)

        relative_scale = np.percentile(np.abs(benign_norms), 50) / (mean_norm + 1e-8)
        if np.isnan(relative_scale):
            relative_scale = 0

        auto_quantile = min(30, max(5, int(relative_scale * 10)))
        raw_q = np.percentile(benign_norms, auto_quantile)
        q = 0.5 * raw_q + 0.5 * mean_norm

        # === 2. base_max_norm 初始估计（限制 std_weight 过大） ===
        adjusted_std_weight = self.std_weight / (1 + std_norm)
        base_max_norm = q + adjusted_std_weight * std_norm

        # === 3. cosine similarity 衰减 ===
        cos_sim = self._compute_cosine_similarity(malicious_update, self._cached_benign_avg)
        decay = 1.0 / (1.0 + np.exp(-5 * (cos_sim - 0.5)))  # sigmoid
        base_max_norm *= decay

        # === 4. distance + 压缩比处理 ===
        euclidean_dist = self._compute_euclidean_distance(malicious_update, self._cached_benign_avg)
        compression_ratio = np.log1p(mean_norm / (euclidean_dist + 1e-8))
        compression_ratio = min(compression_ratio, 1.5)  # 限制压缩比下限
        if euclidean_dist > self.distance_threshold * mean_norm:
            base_max_norm *= compression_ratio
            base_max_norm = max(base_max_norm, self.initial_norm * self.safe_min_ratio)

        # === 5. 正常计划的 max_norm 上限增长控制 ===
        growth_factor = 1 / (1 + np.exp(-(self.round - self.center) / self.scale))
        max_allowed_norm = self.initial_norm * (
                self.safe_min_ratio + (self.safe_max_ratio - self.safe_min_ratio) * growth_factor
        )


        # === 7. 应用上下限 ===
        max_norm = min(base_max_norm, max_allowed_norm)
        max_norm = max(max_norm, self.initial_norm * self.safe_min_ratio)

        # === 8. 打印诊断信息 ===
        print(
            f"[Round {self.round}] cos={cos_sim:.3f} | dist={euclidean_dist:.3f} | base={base_max_norm:.3f} | , {max_allowed_norm:.3f} | final={max_norm:.3f}")

        return max_norm

    def _compute_cosine_similarity(self, a, b):
        flat_a = torch.cat([p.flatten() for p in a.values()])
        flat_b = torch.cat([p.flatten() for p in b.values()])
        return torch.nn.functional.cosine_similarity(flat_a, flat_b, dim=0).item()

    def _compute_euclidean_distance(self, a, b):
        flat_a = torch.cat([p.flatten() for p in a.values()])
        flat_b = torch.cat([p.flatten() for p in b.values()])
        return torch.norm(flat_a - flat_b, p=2).item()

    def _average_model(self, param_list):
        weights = torch.arange(len(param_list), 0, -1, dtype=torch.float32, device=args.device)
        weights /= weights.sum()
        avg = {}
        for name in param_list[0]:
            avg[name] = sum(weights[i] * param_list[i][name].to(args.device) for i in range(len(param_list)))
        return avg

    def _adjust_attack_direction(self, malicious_params, current_params):
        if len(self.attack_direction_queue) > 2:
            momentum = 0.95
            global_dir = self._average_model(list(self.attack_direction_queue))
            current_dir = {k: malicious_params[k] - current_params[k] for k in current_params}
            blended = {k: 0.7 * current_dir[k] + 0.3 * (momentum * global_dir[k] + 0.05 * current_dir[k]) for k in current_params}
            return blended
        else:
            return {k: malicious_params[k] - current_params[k] for k in current_params}

    def malicious_update(self, current_params, historical_grads, negative_grads, states, states_malicious, t_benign=1,
                         t_malicious=30):
        benign_params, new_states = data_less_local_train(current_params, historical_grads, states, t=t_benign)
        malicious_params, new_states_mal = data_less_local_train(current_params, negative_grads, states_malicious,
                                                                 t=t_malicious)
        avg_benign_model = benign_params if not self.benign_model_queue else self._average_model(
            list(self.benign_model_queue))
        benign_update = {k: avg_benign_model[k] - current_params[k] for k in current_params}
        delta_dict = self._adjust_attack_direction(malicious_params, current_params)
        delta = compute_delta_data_free(current_params, historical_grads, states)
        self.delta_history.append(delta)
        smoothed_delta = sum(self.delta_history) / len(self.delta_history)
        if self.use_adaptive_norm:
            max_norm = self._compute_max_norm(self.benign_model_queue, delta_dict)
        else:
            max_norm = 1.86 #or 1.6
        print(f"[Round {self.round}] δ={smoothed_delta:.3f} | max_norm={max_norm:.3f}")

        final_update = {}
        for name in current_params:
            param = current_params[name]
            delta_tensor = delta_dict[name]
            benign_vec = benign_update[name]
            flat_delta = delta_tensor.flatten()
            flat_benign = benign_vec.flatten()
            cos_sim = torch.nn.functional.cosine_similarity(flat_delta, flat_benign, dim=0)
            alpha = torch.sigmoid(10 * (0.8 - cos_sim))
            proj_scale = torch.dot(flat_delta, flat_benign) / (flat_benign.norm() ** 2 + 1e-8)
            projected = proj_scale * benign_vec
            adaptive_blend = (1 - alpha) * delta_tensor + alpha * projected
            u, _, _ = torch.svd(delta_tensor.view(-1, 1))
            structured_noise = 0.3 * max_norm * (u[:, 0] + 0.1 * torch.randn_like(u[:, 0])).view_as(delta_tensor)
            epsilon = 0.4 * max_norm * (1 - cos_sim.item())
            adversarial_noise = epsilon * torch.sign(delta_tensor)
            _, std = self.update_stats.get(name, (None, torch.zeros_like(delta_tensor)))
            stats_noise = -0.4 * max_norm * std
            residual_noise = 0.1 * max_norm * torch.randn_like(delta_tensor)
            if self.round % 5 == 0:
                structured_noise *= 1.5
                adversarial_noise *= 1.5
                stats_noise *= 1.5
            else:
                structured_noise *= 1.1
                adversarial_noise *= 1.1
                stats_noise *= 1.1
            combined_update = adaptive_blend + structured_noise + adversarial_noise + stats_noise + residual_noise
            combined_update += smoothed_delta * torch.sign(delta_tensor)

            # === SVD rank-k 主扰动增强 ===
            svd_strength = 0.3 * max_norm
            svd_k = 5  # 可调秩
            u, s, _ = torch.svd(delta_tensor.view(-1, 1))
            svd_perturbation = sum((s[i] * u[:, i]) for i in range(min(svd_k, u.shape[1]))).view_as(delta_tensor)
            combined_update += svd_strength * svd_perturbation

            # === 正交扰动注入（与 benign 方向正交）===
            flat_benign = benign_vec.flatten()
            orth_noise = torch.randn_like(delta_tensor)
            orth_noise -= torch.dot(orth_noise.flatten(), flat_benign) / (flat_benign.norm() ** 2 + 1e-8) * benign_vec
            orthogonal_strength = 0.3 * max_norm
            combined_update += orthogonal_strength * orth_noise

            norm = combined_update.norm()
            scaling = torch.tanh(norm / max_norm)
            combined_update = combined_update * scaling
            compress = self._compress_update(
                combined_update,  use_norm_clip=True,max_norm=max_norm
            )

            final_update[name] = param + compress

        self.attack_direction_queue.append(malicious_params)
        self.benign_model_queue.append(benign_params)
        self._cached_benign_avg = None
        self.round += 1
        return final_update, new_states, new_states_mal

    def _compress_update(
        self,
        delta: torch.Tensor,
        use_norm_clip=True,
        max_norm=1.0,
    ):
        if use_norm_clip:
            norm = delta.norm()
            if norm > max_norm:
                delta = delta * (max_norm / norm)
        return delta

    """def _compress_update(
            self,
            delta: torch.Tensor,
            benign_vec: torch.Tensor,
            use_projection=False,
            use_norm_clip=True,
            max_norm=1.0,
    ):
        if use_projection:
            flat_delta = delta.flatten()
            flat_benign = benign_vec.flatten()
            proj_scale = torch.dot(flat_delta, flat_benign) / (flat_benign.norm() ** 2 + 1e-8)
            delta = (proj_scale * benign_vec).view_as(delta)
        if use_norm_clip:
            norm = delta.norm()
            if norm > max_norm:
                delta = delta * (max_norm / norm)
        return delta"""