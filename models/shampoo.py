import torch
from typing import Dict, Tuple
import warnings

from utils.parameters import args_parser

args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


def data_less_local_train(
        current_params: Dict[str, torch.Tensor],
        historical_grads: Dict[str, torch.Tensor],
        states: Dict[str, Dict],
        t: int = args.num_malicious + 5,
        beta: float = 0.95,
        epsilon: float = 1e-6,
        lr: float = 0.01,
        prec_update_freq: int = 5,
        step_counter: int = 0,
        update_threshold: float = 0.1,
        noise_scale: float = 0.01,
        max_update: float = 1.0
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict]]:
    """
    改进版无数据联邦学习客户端本地训练函数（具备容错回退）
    """
    new_states = {}
    updated_params = {}
    for i in range(t):
        for name, param in current_params.items():
            # 初始化状态
            if name not in states:
                states[name] = {
                    'L': None,
                    'R': None,
                    'momentum': torch.zeros_like(param, device=args.device),
                    'preconditioner': None,
                    'grad_buffer': torch.zeros_like(param, device=args.device)
                }
                if param.dim() >= 2:
                    rows = param.shape[0]
                    cols = param.numel() // rows
                    states[name]['L'] = epsilon * torch.eye(rows, device=args.device)
                    states[name]['R'] = epsilon * torch.eye(cols, device=args.device)

            state = states[name]
            hist_grad = historical_grads.get(name, torch.zeros_like(param))
            predicted_grad = hist_grad.clone()

            # 更新梯度缓冲区
            state['grad_buffer'] = beta * state['grad_buffer'] + (1 - beta) * predicted_grad
            effective_grad = state['grad_buffer']

            # 矩阵参数处理
            delta = None
            if param.dim() >= 2 and state['L'] is not None:
                # 鲁棒的梯度重塑
                try:
                    grad_matrix = effective_grad.view(state['L'].shape[0], -1)
                except RuntimeError:
                    grad_matrix = effective_grad.reshape(state['L'].shape[0], -1)
                    warnings.warn(f"强制重塑梯度矩阵 {name} 到 {grad_matrix.shape}")

                # 更新统计量
                state['L'] = beta * state['L'] + (1 - beta) * grad_matrix @ grad_matrix.T
                state['R'] = beta * state['R'] + (1 - beta) * grad_matrix.T @ grad_matrix

                # 动态更新条件判断
                grad_norm = torch.norm(grad_matrix).item()

                # ---- 确保 L_reg / R_reg 在任何情况下都定义 ----
                # 默认使用带较弱正则化的版本，保证后续数值操作不会因未定义变量而崩溃
                L_reg = state['L'] + epsilon * torch.eye(*state['L'].shape, device=args.device)
                R_reg = state['R'] + epsilon * torch.eye(*state['R'].shape, device=args.device)

                # 如果满足更新阈值或轮次，使用增强正则化（保留原意）
                if grad_norm > update_threshold or (prec_update_freq > 0 and step_counter % prec_update_freq == 0):
                    try:
                        # 更强的正则化以提升稳定性（尽量保留原始代码意图）
                        stronger_L_reg = state['L'] + epsilon * (1 + torch.max(state['L'])) * torch.eye(
                            *state['L'].shape, device=args.device)
                        stronger_R_reg = state['R'] + epsilon * (1 + torch.max(state['R'])) * torch.eye(
                            *state['R'].shape, device=args.device)
                        L_reg = stronger_L_reg
                        R_reg = stronger_R_reg
                    except Exception:
                        # 如果计算 stronger_* 出错，保持之前的弱正则化
                        warnings.warn(f"在构建增强正则化时遇到问题，使用弱正则化作为回退。参数: {name}")

                try:
                    # 病态性检测与谱分解
                    cond_L = torch.linalg.cond(L_reg)
                    cond_R = torch.linalg.cond(R_reg)
                    if cond_L > 1e5 or cond_R > 1e5:
                        raise RuntimeError(f"Ill-conditioned matrix detected: cond_L={cond_L}, cond_R={cond_R}")

                    eigvals_L, eigvecs_L = torch.linalg.eigh(L_reg)
                    eigvals_R, eigvecs_R = torch.linalg.eigh(R_reg)

                    # 检查是否特征值过于集中或重复
                    def is_bad_eigvals(eigvals):
                        if eigvals.numel() < 2:
                            return False
                        diffs = eigvals[1:] - eigvals[:-1]
                        repeat_ratio = (diffs.abs() < 1e-5).float().mean().item()
                        return repeat_ratio > 0.5

                    if is_bad_eigvals(eigvals_L) or is_bad_eigvals(eigvals_R):
                        raise RuntimeError("Too many repeated eigenvalues.")

                    eigvals_L = torch.clamp(eigvals_L, min=epsilon * 1e-3)
                    eigvals_R = torch.clamp(eigvals_R, min=epsilon * 1e-3)

                    L_inv_4th = eigvecs_L @ torch.diag(eigvals_L.pow(-0.25)) @ eigvecs_L.T
                    R_inv_4th = eigvecs_R @ torch.diag(eigvals_R.pow(-0.25)) @ eigvecs_R.T

                    state['preconditioner'] = (L_inv_4th, R_inv_4th)

                except Exception as e:
                    # 出现任何数值问题时：发出警告并回退到上一次的 preconditioner（如果有），否则使用安全的单位预条件器
                    #warnings.warn(f"Skip eig/preconditioner update due to numerical issue: {e}. Falling back.")
                    if state.get('preconditioner') is not None:
                        # 使用上一轮保存的 preconditioner（就是“使用上一轮的数据”）
                        pass
                    else:
                        # 若没有上一轮数据，则创建一个安全的 identity-style preconditioner（保持不改变逻辑的前提下避免崩溃）
                        try:
                            rows = state['L'].shape[0]
                            cols = state['R'].shape[0]
                            state['preconditioner'] = (torch.eye(rows, device=args.device),
                                                      torch.eye(cols, device=args.device))
                        except Exception:
                            # 最后保底：仍然不设置 preconditioner（后续检查会跳过应用）
                            state['preconditioner'] = None

                # 应用预条件并注入噪声（如果有 preconditioner）
                if state['preconditioner'] is not None:
                    L_pow, R_pow = state['preconditioner']
                    try:
                        delta = L_pow @ grad_matrix @ R_pow.T
                        delta = delta.view_as(param)
                        # 噪声注入
                        if noise_scale > 0:
                            noise = torch.randn_like(delta) * noise_scale * torch.mean(torch.abs(delta))
                            delta += noise
                    except Exception as e:
                        warnings.warn(f"Failed applying preconditioner for {name}: {e}. Falling back to momentum-like update.")
                        delta = None
            else:
                # 一维参数动量更新或无 precondition 情况
                state['momentum'] = beta * state['momentum'] + effective_grad
                delta = state['momentum']

            # 自适应学习率计算
            if delta is not None:
                if param.dim() >= 2 and state['L'] is not None:
                    try:
                        trace_L = torch.trace(state['L'])
                        trace_R = torch.trace(state['R'])
                        trace_scale = torch.sqrt(trace_L * trace_R) + 1e-8
                        effective_lr = lr / trace_scale.item() ** 0.5
                    except Exception:
                        effective_lr = lr
                else:
                    effective_lr = lr

                updated_param = param - effective_lr * delta

                # 异常更新检测
                try:
                    param_change = torch.norm(updated_param - param).item()
                except Exception:
                    param_change = float('inf')

                if param_change > max_update and t <= 5:
                    # 在低轮次迭代中优化良性更新（回退）
                    updated_param = param - lr * effective_grad  # 回退
            else:
                updated_param = param.clone()

            updated_params[name] = updated_param.detach()

            # 状态深拷贝（保留用于下一轮）
            new_states[name] = {
                'L': state['L'].clone() if state['L'] is not None else None,
                'R': state['R'].clone() if state['R'] is not None else None,
                'momentum': state['momentum'].clone(),
                'preconditioner': (
                    state['preconditioner'][0].clone() if state['preconditioner'] and state['preconditioner'][0] is not None else None,
                    state['preconditioner'][1].clone() if state['preconditioner'] and state['preconditioner'][1] is not None else None
                ) if state['preconditioner'] is not None else None,
                'grad_buffer': state['grad_buffer'].clone()
            }

    return updated_params, new_states
