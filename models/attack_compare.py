import copy
import math
from typing import Optional, List

import torch

from utils.parameters import args_parser
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict, deque

args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

class LittleIsEnoughAttack:
    def __init__(self, z_max: float = 1.75):
        self.z_max = z_max

    def generate_poisoned_update(self, state_dicts):
        """
        生成被污染的模型参数字典（投毒 state_dict）

        Args:
            state_dicts (List[Dict[str, Tensor]]): 来自被控 worker 的参数集合

        Returns:
            Dict[str, Tensor]: 投毒后的参数字典
        """
        poisoned_dict = {}
        param_keys = state_dicts[0].keys()

        for key in param_keys:
            stacked = torch.stack([sd[key].float() for sd in state_dicts])
            mu = torch.mean(stacked, dim=0)
            sigma = torch.std(stacked, dim=0)
            poisoned_param = mu + self.z_max * sigma
            poisoned_dict[key] = poisoned_param

        return poisoned_dict


class MinMaxAttack:
    def __init__(self, perturb_type='unit_vector'):
        self.perturb_type = perturb_type

    def generate_poisoned_update(self, benign_dicts):
        avg_dict = self._average_state_dicts(benign_dicts)
        perturb = self._compute_perturbation(avg_dict)
        max_dist = self._max_pairwise_distance_squared(benign_dicts)

        # Use scaled gamma factor (e.g., 1.5x) to slightly exceed max_dist constraint
        perturb_norm = self._state_dict_norm_squared(perturb)
        gamma = (max_dist / (perturb_norm + 1e-6)) ** 0.5
        gamma *= 1.1  # Scale factor as suggested in paper to increase impact on Multi-Krum

        poisoned = {
            k: avg_dict[k] + gamma * perturb[k]
            for k in avg_dict
        }
        return poisoned

    def _average_state_dicts(self, dicts):
        avg_dict = deepcopy(dicts[0])
        for key in avg_dict:
            avg_dict[key] = torch.mean(torch.stack([d[key].float() for d in dicts]), dim=0)
        return avg_dict

    def _compute_perturbation(self, avg_dict):
        perturb = {}
        for k, v in avg_dict.items():
            if self.perturb_type == 'unit_vector':
                perturb[k] = -v / (v.norm() + 1e-6)
            elif self.perturb_type == 'sign':
                perturb[k] = -torch.sign(v)
            else:
                raise ValueError("Unsupported perturb_type")
        return perturb

    def _state_dict_norm_squared(self, d):
        v = torch.nn.utils.parameters_to_vector([d[k].flatten() for k in d])
        return torch.norm(v).pow(2).item()

    def _max_pairwise_distance_squared(self, dicts):
        max_d = 0.0
        n = len(dicts)
        for i in range(n):
            for j in range(i + 1, n):
                d = self._state_dict_distance_squared(dicts[i], dicts[j])
                if d > max_d:
                    max_d = d
        return max_d

    def _state_dict_distance_squared(self, d1, d2):
        v1 = torch.nn.utils.parameters_to_vector([d1[k].flatten() for k in d1])
        v2 = torch.nn.utils.parameters_to_vector([d2[k].flatten() for k in d2])
        return torch.norm(v1 - v2).pow(2).item()


class MinSumAttack:
    def __init__(self, perturb_type='unit_vector'):
        self.perturb_type = perturb_type

    def generate_poisoned_update(self, benign_dicts):
        avg_dict = self._average_state_dicts(benign_dicts)
        perturb = self._compute_perturbation(avg_dict)
        max_sum = self._max_sum_distance_squared(benign_dicts)

        perturb_norm = self._state_dict_norm_squared(perturb)
        gamma = (max_sum / (len(benign_dicts) * perturb_norm + 1e-6)) ** 0.5

        poisoned = {
            k: avg_dict[k] + gamma * perturb[k]
            for k in avg_dict
        }
        return poisoned

    def _average_state_dicts(self, dicts):
        avg_dict = deepcopy(dicts[0])
        for key in avg_dict:
            avg_dict[key] = torch.mean(torch.stack([d[key].float() for d in dicts]), dim=0)
        return avg_dict

    def _compute_perturbation(self, avg_dict):
        perturb = {}
        for k, v in avg_dict.items():
            if self.perturb_type == 'unit_vector':
                perturb[k] = -v / (v.norm() + 1e-6)
            elif self.perturb_type == 'sign':
                perturb[k] = -torch.sign(v)
            else:
                raise ValueError("Unsupported perturb_type")
        return perturb

    def _state_dict_norm_squared(self, d):
        v = torch.nn.utils.parameters_to_vector([d[k].flatten() for k in d])
        return torch.norm(v).pow(2).item()

    def _sum_distance_to_all(self, d1, dicts):
        return sum([self._state_dict_distance_squared(d1, d2) for d2 in dicts])

    def _max_sum_distance_squared(self, dicts):
        return max([self._sum_distance_to_all(d, dicts) for d in dicts])

    def _state_dict_distance_squared(self, d1, d2):
        v1 = torch.nn.utils.parameters_to_vector([d1[k].flatten() for k in d1])
        v2 = torch.nn.utils.parameters_to_vector([d2[k].flatten() for k in d2])
        return torch.norm(v1 - v2).pow(2).item()

import torch
from copy import deepcopy

class FangAttack:
    def __init__(self, epsilon=1e-5, max_iter=20, tol=1e-5, device=args.device):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.device = device

    def generate_poisoned_models(self, global_model_dict, malicious_model_dicts, num_total_clients, num_malicious):
        s_hat = self._estimate_update_direction(global_model_dict, malicious_model_dicts)

        lambda_max = self._search_lambda(
            s_hat, global_model_dict,
            malicious_model_dicts, num_total_clients, num_malicious
        )

        poisoned_model = {
            k: global_model_dict[k] + lambda_max * s_hat[k]
            for k in global_model_dict
        }

        return poisoned_model  # 单个共享的恶意模型

    def _estimate_update_direction(self, global_model_dict, malicious_model_dicts):
        s_hat = {}
        for k in global_model_dict:
            diffs = [
                client[k].to(self.device) - global_model_dict[k].to(self.device)
                for client in malicious_model_dicts
            ]
            s_hat[k] = torch.mean(torch.stack(diffs), dim=0)
        return s_hat

    def _search_lambda(self, s_hat, global_model_dict, malicious_model_dicts, n, f):
        lambda_low = 0.0
        lambda_high = 10.0
        best_lambda = 0.0

        for _ in range(self.max_iter):
            lambda_mid = (lambda_low + lambda_high) / 2.0
            poisoned_update = {
                k: global_model_dict[k] + lambda_mid * s_hat[k]
                for k in global_model_dict
            }

            updates = malicious_model_dicts + [poisoned_update for _ in range(f)]

            if self._krum_selects_poisoned(updates, f):
                best_lambda = lambda_mid
                lambda_low = lambda_mid
            else:
                lambda_high = lambda_mid

            if lambda_high - lambda_low < self.tol:
                break

        return best_lambda

    def _krum_selects_poisoned(self, updates, f):
        distances = []
        for i in range(len(updates)):
            dists = []
            for j in range(len(updates)):
                if i != j:
                    d = self._distance_squared(updates[i], updates[j])
                    dists.append(d)
            dists.sort()
            score = sum(dists[:len(updates) - f - 2])
            distances.append(score)

        selected = distances.index(min(distances))
        return selected >= len(updates) - f

    def _distance_squared(self, d1, d2):
        v1 = torch.nn.utils.parameters_to_vector(
            [d1[k].float().flatten().to(self.device) for k in sorted(d1)]
        )
        v2 = torch.nn.utils.parameters_to_vector(
            [d2[k].float().flatten().to(self.device) for k in sorted(d2)]
        )
        return torch.norm(v1 - v2).pow(2).item()



class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, outputs, labels):
        probs = F.softmax(outputs, dim=1)
        correct_probs = probs[torch.arange(probs.size(0)), labels]
        mask = torch.ones_like(probs, dtype=torch.bool)
        mask[torch.arange(probs.size(0)), labels] = False
        other_probs = probs[mask].view(probs.size(0), -1)
        max_other_probs, _ = torch.max(other_probs, dim=1)
        loss = torch.clamp(correct_probs - max_other_probs, min=0)
        return loss.mean()


class ExponentialSmoothing:
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.s1 = self.s2 = None

    def update(self, global_model):
        if self.s1 is None:
            self.s1 = OrderedDict((k, v.clone()) for k, v in global_model.items())
            self.s2 = OrderedDict((k, v.clone()) for k, v in global_model.items())
        else:
            for key in global_model.keys():
                self.s1[key] = self.alpha * global_model[key] + (1 - self.alpha) * self.s1[key]
                self.s2[key] = self.alpha * self.s1[key] + (1 - self.alpha) * self.s2[key]

    def predict(self):
        if self.s1 is None or self.s2 is None:
            raise ValueError("Exponential smoothing values are not initialized.")
        predicted_model = OrderedDict()
        for key in self.s1.keys():
            predicted_model[key] = ((2 - self.alpha) / (1 - self.alpha)) * self.s1[key] - (1 / (1 - self.alpha)) * self.s2[key]
        return predicted_model


def estimate_certified_radius(historical_updates):
    if not historical_updates:
        return 1.0
    std_dev = OrderedDict()
    for key in historical_updates[0].keys():
        updates = torch.stack([u[key] for u in historical_updates])
        std_dev[key] = torch.std(updates, dim=0)
    certified_radius = sum(torch.norm(v, p=2).item()**2 for v in std_dev.values())**0.5 * 3
    return certified_radius


def project_update(malicious_update, global_model, certified_radius):
    distance = sum(torch.norm(malicious_update[k] - global_model[k], p=2).item()**2 for k in malicious_update.keys())**0.5
    if distance > certified_radius:
        scale = certified_radius / (distance + 1e-10)
        return OrderedDict((k, global_model[k] + scale * (malicious_update[k] - global_model[k])) for k in malicious_update.keys())
    return malicious_update


def optimize_perturbation(model, inputs, labels, lambda_, num_steps=100, lr=0.01):
    delta_p = torch.zeros_like(inputs, requires_grad=True)
    optimizer = optim.SGD([delta_p], lr=lr)
    criterion = HingeLoss()

    for step in range(num_steps):
        perturbed_outputs = model(inputs + delta_p)
        loss_perturb = lambda_ * torch.norm(delta_p, p=2)
        loss_model = criterion(perturbed_outputs, labels)
        loss = loss_perturb + loss_model

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return delta_p


def generate_malicious_model(
    reference_model: OrderedDict,
    model_fn,  # 返回一个与 reference_model 结构匹配的模型实例
    malicious_clients_data_list,
    threshold: float,
    lambda_: float,
    device,
    historical_updates: list
) -> OrderedDict:
    """
    使用 I-FMPA 生成一个完整的“恶意模型”参数字典，可直接 load 到模型中。

    参数:
        reference_model: 预测的参考模型参数（OrderedDict）
        model_fn: 返回模型实例的函数
        malicious_clients_data_list: List of (train_loader, val_loader) tuples
        threshold: 验证精度阈值，用于 early stopping
        lambda_: 扰动控制系数
        device: CUDA 或 CPU
        historical_updates: 历史全局模型参数（用于估算扰动边界）

    返回:
        OrderedDict: 恶意模型的完整参数，可直接 load 到模型
    """

    # 初始化模型
    model = model_fn().to(device)
    model.load_state_dict(copy.deepcopy(reference_model))
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = HingeLoss()
    certified_radius = estimate_certified_radius(historical_updates)

    for epoch in range(50):
        for train_loader, _ in malicious_clients_data_list:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                delta_p = optimize_perturbation(model, inputs, labels, lambda_)
                perturbed_inputs = inputs + delta_p
                outputs = model(perturbed_inputs)

                loss_clean = criterion(outputs, labels)
                loss_perturb = lambda_ * torch.norm(delta_p, p=2)
                loss = loss_clean + loss_perturb

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 计算平均验证准确率
        model.eval()
        total_correct, total_samples = 0, 0
        with torch.no_grad():
            for _, val_loader in malicious_clients_data_list:
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)

        acc = total_correct / total_samples
        print(f"[I-FMPA] Epoch {epoch + 1}: Val Acc = {acc:.4f}")
        if acc <= threshold:
            break

        model.train()

    # 获取恶意模型的参数，并投影（以防越界）
    final_state = model.state_dict()
    projected_state = project_update(final_state, reference_model, certified_radius)

    return projected_state

class FedGhostAttacker:
    """
    FedGhost (practical):
    - 预测下一轮全局梯度：用最近 N 轮 ΔW, ΔG 做多割线最小二乘，Δg_pred = Y @ ( (S^T S)^-1 S^T s_last )
      使得在 span(S) 上近似满足 y ≈ H s，并据此预测 H s_last（论文式(8)的实用化形态）。[FedGhost Alg.1 思路]
    - 生成两类恶意梯度：concealed（贴近预测方向）与 sacrificial（反向放大）。[式(11)(12)]
    - 反馈自适应：基于上一轮攻击后全局梯度与恶意梯度均值的余弦相似度 cs，做 reward/penalty 调 γ，夹在 [γ_min, γ_max]。[式(13)+Alg.2]
    参考：:contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}
    """

    def __init__(self, model_fn, num_malicious=5, window_size=10, eta=1.0, device='cpu',
                 gamma_min=1e-5, gamma_max=5.0, cmin=0.5, sacrificial_scale=10.0,
                 topk_ratio=1.0,  # 论文里只在大幅值坐标上加扰动；1.0=全部坐标，<1选TopK  :contentReference[oaicite:10]{index=10}
                 ):
        self.model_fn = model_fn
        self.num_malicious = num_malicious
        self.device = device
        self.eta = eta

        # 历史序列（扁平向量）
        self.w_hist = deque(maxlen=window_size+1)  # 存 w_t
        self.g_hist = deque(maxlen=window_size+1)  # 存 g_t（若拿不到则用 Δw/eta 近似）
        self.window_size = window_size

        # 自适应 γ
        self.gamma = 1.0
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.cmin = cmin
        self.sacrificial_scale = sacrificial_scale

        # 只在大幅值坐标上加扰动
        self.topk_ratio = topk_ratio

        # 记录最新反馈
        self.last_after_attack_global_grad = None
        self.last_cs = None
        self._prev_beta = None

    # ------------ 工具：flatten/unflatten ------------
    def _flatten(self, state_dict):
        flats = []
        for p in state_dict.values():
            flats.append(p.view(-1).to(self.device))
        return torch.cat(flats)

    def _unflatten_like(self, flat_vec, like_state):
        out = OrderedDict()
        offset = 0
        for k, v in like_state.items():
            num = v.numel()
            out[k] = flat_vec[offset:offset+num].view_as(v).to(v.dtype).to(v.device)
            offset += num
        return out

    # ------------ 输入记录 ------------
    def add_global(self, model_state_dict, global_grad_state_dict=None):
        """
        在每轮开始/结束时记录服务器广播：
        - model_state_dict: 全局模型参数 w_t
        - global_grad_state_dict: 服务器的全局梯度 g_t（若拿不到，可传 None）
        """
        w = self._flatten(model_state_dict).detach().to(self.device)
        self.w_hist.append(w)

        if global_grad_state_dict is not None:
            g = self._flatten(global_grad_state_dict).detach().to(self.device)
            self.g_hist.append(g)
        else:
            # 若拿不到 g_t，则用 Δw/eta 近似 g_t（与你原先一致，作为退化兜底）
            if len(self.w_hist) >= 2:
                g_approx = (self.w_hist[-1] - self.w_hist[-2]) / max(self.eta, 1e-12)
                self.g_hist.append(g_approx)

    def add_after_attack_global_grad(self, global_grad_state_dict):
        """供每轮聚合后（含我们的恶意上传）喂入 g_after，用于反馈调参。"""
        self.last_after_attack_global_grad = self._flatten(global_grad_state_dict).detach().to(self.device)

    # ------------ 基于 (ΔW, ΔG) 的 Δg 预测（实用版 Alg.1） ------------
    def _predict_next_global_grad(self):
        """
        用最近 N 轮 S=[Δw]、Y=[Δg]，对 s_last 做最小二乘投影：
            β = (S^T S + λI)^{-1} S^T s_last
            Δg_pred = Y β
        然后 g_pred = g_{t-1} + Δg_pred
        """
        m = min(self.window_size, len(self.w_hist)-1, len(self.g_hist)-1)
        if m < 2:
            raise ValueError("需要至少2个 ΔW/ΔG 历史条目以预测下一轮梯度。")

        # 构造 ΔW, ΔG（列堆叠为 d×m）
        S_cols, Y_cols = [], []
        for i in range(-m, 0):
            S_cols.append((self.w_hist[i] - self.w_hist[i-1]).unsqueeze(1))
            Y_cols.append((self.g_hist[i] - self.g_hist[i-1]).unsqueeze(1))
        S = torch.cat(S_cols, dim=1)  # d×m
        Y = torch.cat(Y_cols, dim=1)  # d×m

        s_last = self.w_hist[-1] - self.w_hist[-2]
        g_last = self.g_hist[-1]

        # 计算 β（Tikhonov 稳定）
        StS = S.T @ S
        lam = 1e-6 * torch.trace(StS) / max(m,1)  # 小正则
        I = torch.eye(m, device=self.device, dtype=StS.dtype)
        rhs = S.T @ s_last
        try:
            beta = torch.linalg.solve(StS + lam * I, rhs)
        except (torch._C._LinAlgError, RuntimeError) as e:
            # 仅在“上一轮 beta 可用且长度匹配且元素有穷”时回退；否则继续抛出异常
            if (self._prev_beta is not None
                    and self._prev_beta.numel() == m
                    and torch.isfinite(self._prev_beta).all()):
                beta = self._prev_beta.to(device=self.device, dtype=StS.dtype)
                print(f"[FedGhost][warn] use previous beta due to solve failure: {e}")
            else:
                raise

        # 成功求得（或回退）后，缓存本轮 beta 以备下轮回退使用
        self._prev_beta = beta.detach().clone()

        delta_g_pred = Y @ beta                           # d
        g_pred = g_last + delta_g_pred                    # 预测下一轮全局梯度 g_pre

        return g_pred, delta_g_pred

    # ------------ 只在 top-|Δg_pred| 坐标加扰动 ------------
    def _mask_topk(self, vec):
        if self.topk_ratio >= 1.0:
            return torch.ones_like(vec, dtype=torch.bool)
        d = vec.numel()
        k = max(1, int(d * self.topk_ratio))
        topk = torch.topk(vec.abs(), k, sorted=False).indices
        mask = torch.zeros(d, dtype=torch.bool, device=vec.device)
        mask[topk] = True
        return mask

    # ------------ 自适应 γ（基于反馈 cs 与阈值 Cmin） ------------
    def _adapt_gamma(self, malicious_stack, after_attack_global_grad):
        """
        cs = cos( g_after , mean(malicious) ). 若 cs >= Cmin -> reward(增大γ)
        否则 -> penalty(减小γ)。论文式(13)+Alg.2。:contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}
        """
        mbar = malicious_stack.mean(dim=1)  # d
        a = after_attack_global_grad
        cs = torch.dot(a, mbar) / (a.norm()*mbar.norm() + 1e-12)
        self.last_cs = cs.item()

        # 简单线性步长（也可用比例因子/动量）
        if cs >= self.cmin:
            self.gamma *= 1.25
        else:
            self.gamma *= 0.75
        self.gamma = float(max(self.gamma_min, min(self.gamma, self.gamma_max)))

    # ------------ 生成恶意梯度 ------------
    def generate_malicious_gradients(self, current_global_model, sacrifice_ratio=0.2):
        """
        返回 num_malicious 个“替代本地梯度”的 OrderedDict（与 current_global_model 同 shape）。
        - 需要事先调用 add_global(...) 若干次，凑足历史。
        - 若上一轮聚合后的 g_after 已喂入（add_after_attack_global_grad），会先做一次 γ 自适应。
        """
        # 1) 预测下一轮全局梯度 g_pre（论文 A.1 / Alg.1 思路）:contentReference[oaicite:13]{index=13}
        g_pre, delta_g_pred = self._predict_next_global_grad()

        # 2) 基于 Δg_pred 的单位方向，做 concealed / sacrificial（式(11)(12)）:contentReference[oaicite:14]{index=14}
        #    只在 top-|Δg_pred| 坐标上加扰动（论文 remark）:contentReference[oaicite:15]{index=15}
        unit = delta_g_pred / (delta_g_pred.norm() + 1e-12)
        mask = self._mask_topk(delta_g_pred)
        dir_masked = unit * mask.to(unit.dtype)

        # 若有上一轮反馈，先调一次 γ
        if self.last_after_attack_global_grad is not None:
            # 为计算 cs，先用当前 γ 构造一个“临时的”恶意堆栈
            d = unit.numel()
            num_sacrifice = int(self.num_malicious * sacrifice_ratio)
            tmp = []
            for i in range(self.num_malicious):
                scale = self.sacrificial_scale if i < num_sacrifice else 1.0
                grad_flat = (-scale if i < num_sacrifice else 1.0) * self.gamma * dir_masked
                tmp.append(grad_flat.view(d, 1))
            tmp_stack = torch.cat(tmp, dim=1)  # d×M
            self._adapt_gamma(tmp_stack, self.last_after_attack_global_grad)  # 更新 γ

        # 最终恶意梯度
        d = unit.numel()
        num_sacrifice = int(self.num_malicious * sacrifice_ratio)
        malis = []
        for i in range(self.num_malicious):
            scale = self.sacrificial_scale if i < num_sacrifice else 1.0
            sign = -1.0 if i < num_sacrifice else 1.0
            grad_flat = sign * self.gamma * dir_masked  # d
            malis.append(grad_flat.view(d, 1))
        M = torch.cat(malis, dim=1)  # d×M

        # 3) 还原成 state_dict 形状
        like = OrderedDict((k, v.detach().clone().to(self.device)) for k, v in current_global_model.items())
        malicious = []
        for j in range(self.num_malicious):
            malicious.append(self._unflatten_like(M[:, j], like))

        return malicious



class PoisonedFLAttacker:
    """
    通用版 PoisonedFL（自适应 beta/eps）：
    - 固定符号向量 s（±1），跨轮方向一致性。
    - k_t = alpha_t * v_t，alpha_t = c_t * ||g_{t-1}||；v_t 基于 |g_{t-1} - scale * (k_{t-1}·s)| 的幅度（可选分层归一）。
    - 每 e 轮做一次二项检验：若过去 e 轮总更新 ΔW 的符号与 s 的一致性 < Bin(d,0.5) 右尾阈值（align_p），
      则“失败”：c 衰减、beta/eps 增加；相反连续成功则轻增 c、并降低 beta/eps。
    - 若服务器聚合用 w <- w - lr*grad，则 as_gradient=True（把更新方向自动翻到 -s 以实现沿 +s 推动）。

    推荐初始化（Dirichlet α=0.5）：
        attacker = PoisonedFLAttacker(
            num_malicious=args.num_malicious, device=args.device, as_gradient=True,
            c0=10.0, e=25, rho=0.9, c_min=2.0, align_p=0.95,
            layered_v=True, warmup_rounds=5,
            beta0=0.10, eps0=1e-2
        )
    """

    def __init__(self,
                 num_malicious: int = 5,
                 device: str | torch.device = 'cpu',
                 *,
                 # 论文对齐的幅值调度
                 c0: float = 8.0,
                 e: int = 50,
                 rho: float = 0.7,
                 c_min: float = 0.5,
                 align_p: float = 0.99,
                 # “上传梯度/增量”的方向约定
                 as_gradient: bool = True,
                 # 分层 v_t（更拟真）
                 layered_v: bool = False,
                 warmup_rounds: int = 0,
                 # 连续成功轻增长
                 grow: float = 1.1,
                 c_max: float = 64.0,
                 # —— 通用自适应伪装参数（不依赖 defense_type）——
                 beta0: float = 0.10,   # 初始“向群体混合”比例
                 eps0: float = 1e-2,    # 初始“恶意间多样性”强度
                 beta_min: float = 0.0,
                 beta_max: float = 0.30,
                 eps_min: float = 0.0,
                 eps_max: float = 2e-2,
                 beta_inc: float = 1.25,  # 失败时 *1.25
                 beta_dec: float = 0.90,  # 成功时 *0.90
                 eps_inc: float = 1.25,
                 eps_dec: float = 0.90):
        self.num_malicious = int(num_malicious)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        # 历史全局模型
        self.w_hist: deque[torch.Tensor] = deque(maxlen=e + 2)
        self.round_idx: int = 0

        # c_t 调度与检验
        self.c: float = float(c0)
        self.c0: float = float(c0)
        self.e: int = int(e)
        self.rho: float = float(rho)
        self.c_min: float = float(c_min)
        self.align_p: float = float(align_p)

        self.success_streak: int = 0
        self.grow: float = float(grow)
        self.c_max: float = float(c_max)

        # 语义/工程选项
        self.as_gradient: bool = bool(as_gradient)
        self.layered_v: bool = bool(layered_v)
        self.warmup_rounds: int = int(warmup_rounds)

        # 自适应伪装（通用，不看 defense）
        self.beta: float = float(beta0)
        self.eps: float = float(eps0)
        self.beta_min: float = float(beta_min)
        self.beta_max: float = float(beta_max)
        self.eps_min: float = float(eps_min)
        self.eps_max: float = float(eps_max)
        self.beta_inc: float = float(beta_inc)
        self.beta_dec: float = float(beta_dec)
        self.eps_inc: float = float(eps_inc)
        self.eps_dec: float = float(eps_dec)

        # 固定 s 与上一轮恶意
        self.s: Optional[torch.Tensor] = None
        self.last_mal: Optional[torch.Tensor] = None

    # ---------- 工具 ----------
    def _flatten(self, sd: OrderedDict) -> torch.Tensor:
        flats = [p.detach().to(self.device).float().view(-1) for p in sd.values()]
        return torch.cat(flats, dim=0)

    def _unflatten_like(self, flat: torch.Tensor, like: OrderedDict) -> OrderedDict:
        out = OrderedDict()
        off = 0
        for k, v in like.items():
            n = v.numel()
            out[k] = flat[off:off+n].view_as(v).to(v.dtype).to(v.device)
            off += n
        return out

    @staticmethod
    def _rt_binom_thr(d: int, p: float) -> int:
        # 正态近似右尾阈值；若 p 未在表中，默认 0.99
        z_map = {
            0.90: 1.2815515655446004,
            0.95: 1.6448536269514722,
            0.99: 2.3263478740408408,
        }
        z = z_map.get(p, 2.3263478740408408)
        mu, sigma = 0.5 * d, math.sqrt(0.25 * d)
        return int(math.ceil(mu + z * sigma))

    def add_global(self, global_state: OrderedDict):
        w = self._flatten(global_state)
        self.w_hist.append(w)
        self.round_idx += 1

    # ---------- v_t ----------
    def _vt_flat(self, g_prev: torch.Tensor, last_mal: Optional[torch.Tensor]) -> torch.Tensor:
        eps = 1e-12
        if last_mal is None or last_mal.numel() == 0:
            base = g_prev.abs()
        else:
            # FIX-1: 去掉 as_gradient 的符号影响：使用 (last_mal / sgn) 做参照
            sgn = -1.0 if self.as_gradient else 1.0
            ref = last_mal / sgn                    # ≈ k_{t-1}·s
            scale = g_prev.norm() / (ref.norm() + eps)
            base = (g_prev - scale * ref).abs()
        return base / (base.norm() + eps)

    def _vt_layered(self, g_prev: torch.Tensor, last_mal: Optional[torch.Tensor], like: OrderedDict) -> torch.Tensor:
        eps = 1e-12
        vt_list = []
        off = 0
        if last_mal is None or last_mal.numel() == 0:
            for _, v in like.items():
                n = v.numel()
                seg = g_prev[off:off+n]
                vt_list.append(seg.abs() / (seg.abs().norm() + eps))
                off += n
        else:
            # FIX-1: 分段同理，使用 (last_mal / sgn)
            sgn = -1.0 if self.as_gradient else 1.0
            ref = last_mal / sgn
            for _, v in like.items():
                n = v.numel()
                seg = g_prev[off:off+n]
                ref_seg = ref[off:off+n]
                scale = seg.norm() / (ref_seg.norm() + eps)
                base = (seg - scale * ref_seg).abs()
                vt_list.append(base / (base.norm() + eps))
                off += n
        vt = torch.cat(vt_list, dim=0)
        return vt / (vt.norm() + eps)

    # ---------- 自适应调度：每 e 轮 ----------
    def _schedule_and_adapt(self):
        if len(self.w_hist) < self.e + 1 or self.s is None:
            return

        w_last = self.w_hist[-1]
        w_e_ago = self.w_hist[-(self.e + 1)]
        total_update = w_last - w_e_ago

        # 避免 0 的不确定符号：用 w_last 兜底
        zero_mask = (total_update == 0)
        if zero_mask.any():
            total_update = torch.where(zero_mask, w_last, total_update)

        sign_total = torch.sign(total_update)
        aligned = (sign_total == self.s).sum().item()
        d = int(sign_total.numel())
        thr = self._rt_binom_thr(d, self.align_p)

        if (self.round_idx % self.e) == 0:
            if aligned < thr:
                # 失败：c 衰减 + 增强伪装
                self.c = max(self.c * self.rho, self.c_min)
                self.success_streak = 0
                self.beta = min(self.beta * self.beta_inc, self.beta_max)
                self.eps  = min(self.eps  * self.eps_inc,  self.eps_max)
            else:
                # 成功：累计并适度增益，弱化伪装
                self.success_streak += 1
                if self.success_streak >= 3:
                    self.c = min(self.c * self.grow, self.c_max)
                    self.success_streak = 0
                self.beta = max(self.beta * self.beta_dec, self.beta_min)
                self.eps  = max(self.eps  * self.eps_dec,  self.eps_min)

    # ---------- 生成恶意更新 ----------
    def generate_malicious_gradients(self, current_global_model: OrderedDict) -> List[OrderedDict]:
        like = OrderedDict((k, v.detach().clone().to(self.device)) for k, v in current_global_model.items())
        d = sum(v.numel() for v in like.values())

        # 初始化 s
        if self.s is None or self.s.numel() != d:
            rnd = torch.randint(0, 2, (d,), device=self.device, dtype=torch.int32)
            self.s = (2 * rnd - 1).to(torch.float32)

        # FIX-3: 暖启动或历史不足：返回 0 更新（不抛错）
        if self.round_idx <= max(1, self.warmup_rounds) or len(self.w_hist) < 2:
            zero = torch.zeros(d, device=self.device, dtype=torch.float32)
            return [self._unflatten_like(zero, like) for _ in range(self.num_malicious)]

        # g_{t-1}
        g_prev = self.w_hist[-1] - self.w_hist[-2]

        # v_t
        if self.layered_v:
            v_t = self._vt_layered(g_prev, self.last_mal, like)
        else:
            v_t = self._vt_flat(g_prev, self.last_mal)

        # FIX-2: 先检验并自适应，再计算 alpha_t 使用当轮 c_t
        self._schedule_and_adapt()
        alpha_t = self.c * (g_prev.norm() + 1e-12)

        # 方向约定
        sgn = -1.0 if self.as_gradient else 1.0
        target_sign = sgn * self.s

        # 核心恶意向量
        m_flat = sgn * (alpha_t * v_t * self.s)

        # 通用伪装：向群体混合
        if self.beta > 0.0:
            mix_target = (-g_prev) if self.as_gradient else g_prev
            m_flat = (1.0 - self.beta) * m_flat + self.beta * mix_target

        # 保持与 target_sign 的符号一致（跨轮一致性）
        m_flat = torch.sign(target_sign) * m_flat.abs()

        # 记录到下一轮
        self.last_mal = m_flat.detach().clone()

        # 恶意间微多样性
        malicious: List[OrderedDict] = []
        if self.eps > 0.0:
            base_scale = (m_flat.norm() / (math.sqrt(m_flat.numel()) + 1e-12)).item()
        for _ in range(self.num_malicious):
            if self.eps > 0.0:
                noise = torch.randn_like(m_flat) * (self.eps * base_scale)
                m_i = m_flat + noise
                m_i = torch.sign(target_sign) * m_i.abs()
            else:
                m_i = m_flat
            malicious.append(self._unflatten_like(m_i, like))

        return malicious


