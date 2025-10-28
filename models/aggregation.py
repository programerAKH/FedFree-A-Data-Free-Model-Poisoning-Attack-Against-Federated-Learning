import copy
from typing import Dict, List, Any, Tuple
import numpy as np

import torch
from collections import OrderedDict

from numpy import floating
from torch import Tensor

from models.test import test_img
from utils.countDict import count_malicious_dicts
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
"""
联邦学习的防御策略,包括FedAvg(2016),Multi-krum(2017),Bulyan(2018),RFA(2022),CC(2021),RoFL(2023)
"""


#传入的参数w是所有选中的本地客户端训练完后得到的数组，每个元素是一个client训练后的参数字典
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])#将参数中的第一个元素深拷贝（原始对象修改后不会影响深拷贝对象）到w_avg中.
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))#计算平均值并返回
    return w_avg

def krum(client_updates, f):
    """
    Krum 算法实现
    :param client_updates: 客户端更新列表，每个更新是一个参数字典（state_dict）
    :param f: 恶意客户端数量的上限
    :return: 聚合后的参数字典
    """
    n = len(client_updates)  # 客户端数量
    num_selected = n - f - 2  # 选择的最近更新的数量

    # 将参数字典转换为张量
    client_tensors = [torch.cat([param.view(-1) for param in update.values()]) for update in client_updates]

    # 计算每对客户端更新之间的欧氏距离
    distances = []
    for i in range(n):
        dist = []
        for j in range(n):
            if i != j:
                dist.append(torch.norm(client_tensors[i] - client_tensors[j], p=2).item())
        dist.sort()  # 按距离排序
        distances.append(sum(dist[:num_selected]))  # 选择最近的 num_selected 个距离

    # 选择总距离最小的客户端更新
    selected_index = distances.index(min(distances))
    return client_updates[selected_index]

def multi_krum(client_updates, f, m,param_malicious=None):
    """
    Multi-Krum 算法实现
    :param client_updates: 客户端更新列表，每个更新是一个参数字典（state_dict）
    :param f: 恶意客户端数量的上限
    :param m: 选择的客户端更新数量
    :param param_malicious:恶意客户端参数
    :return: 聚合后的参数字典,被选中的恶意客户端数量
    """
    n = len(client_updates)  # 客户端数量
    num_selected = n - f - 2  # 选择的最近更新的数量
    aggregated_update = copy.deepcopy(client_updates[0])

    # 将参数字典转换为张量
    client_tensors = [torch.cat([param.view(-1) for param in update.values()]) for update in client_updates]
    selected_malicious = 0
    # 计算每对客户端更新之间的欧氏距离
    distances = []
    for i in range(n):
        dist = []
        for j in range(n):
            if i != j:
                dist.append(torch.norm(client_tensors[i] - client_tensors[j], p=2).item())
        dist.sort()  # 按距离排序
        distances.append(sum(dist[:num_selected]))  # 选择最近的 num_selected 个距离

    # 选择总距离最小的 m 个客户端更新
    selected_indices = sorted(range(n), key=lambda x: distances[x])[:m]
    selected_updates = [client_updates[i] for i in selected_indices]
    # 对每个参数取平均值
    for key in client_updates[0].keys():
        aggregated_update[key] = torch.mean(torch.stack([update[key] for update in selected_updates]), dim=0)
    if param_malicious is not None:
        selected_malicious = count_malicious_dicts(param_malicious, selected_updates)
    return aggregated_update,selected_malicious
def trimmed_mean(updates, beta):
    """
    Trimmed Mean 算法实现
    :param updates: 客户端更新列表，每个更新是一个参数字典
    :param beta: 去掉的最大和最小值的数量
    :return: 聚合后的更新
    """
    aggregated_update = OrderedDict()

    # 对每个参数进行 Trimmed Mean 聚合
    for key in updates[0].keys():
        stacked_params = torch.stack([update[key] for update in updates])
        sorted_params, _ = torch.sort(stacked_params, dim=0)
        trimmed_params = sorted_params[beta:-beta]  # 去掉最大和最小的 beta 个值
        aggregated_update[key] = torch.mean(trimmed_params, dim=0)
    return aggregated_update


def bulyan(client_updates, f, m, beta,param_malicious=None):
    """
    Bulyan 算法实现
    :param client_updates: 客户端更新列表，每个更新是一个参数字典
    :param f: 恶意客户端数量的上限
    :param m: 筛选阶段选择的客户端更新数量
    :param beta: Trimmed Mean 中去掉的最大和最小值的数量
    :param param_malicious:恶意客户端生成的参数字典
    :return: 聚合后的更新
    """
    n = len(client_updates)  # 客户端数量
    num_selected = n - f - 2  # 选择的最近更新的数量
    distances = []

    # 将参数字典转换为张量
    client_tensors = [torch.cat([param.view(-1) for param in update.values()]) for update in client_updates]
    selected_malicious = 0
    # 计算每对客户端更新之间的欧氏距离
    for i in range(n):
        dist = []
        for j in range(n):
            if i != j:
                dist.append(torch.norm(client_tensors[i] - client_tensors[j], p=2).item())
        dist.sort()  # 按距离排序
        distances.append(sum(dist[:num_selected]))  # 选择最近的 num_selected 个距离

    # 选择总距离最小的 m 个客户端更新
    selected_indices = sorted(range(n), key=lambda x: distances[x])[:m]
    selected_updates = [client_updates[i] for i in selected_indices]
    # 聚合阶段：使用 Trimmed Mean 聚合候选更新
    aggregated_update = trimmed_mean(selected_updates, beta)
    if param_malicious is not None:
        selected_malicious = count_malicious_dicts(param_malicious, selected_updates)
    return aggregated_update,selected_malicious

def median(client_params_list):
    """
    联邦学习中基于中位数的参数聚合算法
    输入:
        client_params_list (list of dict): 客户端参数列表，每个元素为参数字典（键为参数名，值为张量）
    返回:
        aggregated_params (dict): 聚合后的参数字典，键为参数名，值为中位数张量
    """
    aggregated_params = {}

    # 检查输入有效性
    if not client_params_list:
        raise ValueError("客户端参数列表不能为空")

    # 获取所有参数名（假设所有客户端参数结构一致）
    param_names = client_params_list[0].keys()

    for param_name in param_names:
        try:
            # 收集所有客户端的当前参数张量，并确保在同一设备
            param_tensors = []
            for client_params in client_params_list:
                tensor = client_params[param_name].detach().cpu()  # 统一移动到CPU
                param_tensors.append(tensor)

            # 堆叠张量（维度: [num_clients, ...]）
            stacked = torch.stack(param_tensors, dim=0)

            # 计算中位数（沿客户端维度）
            median_tensor = torch.median(stacked, dim=0).values  # 使用.values获取中位数

            aggregated_params[param_name] = median_tensor

        except KeyError:
            raise RuntimeError(f"参数名 {param_name} 在部分客户端中缺失")
        except RuntimeError as e:
            raise RuntimeError(f"参数 {param_name} 聚合失败: {str(e)}")

    return aggregated_params


class FedGT:
    def __init__(self, num_groups: int = 3, overlap_ratio: float = 0.6):
        """
        FedGT防御初始化
        :param num_groups: 分组数量
        :param overlap_ratio: 客户端分配到各组的比例
        """
        self.num_groups = num_groups
        self.overlap_ratio = overlap_ratio

    def create_group_matrix(self, num_users: int) -> np.ndarray:
        """
        创建客户端-组分配矩阵
        :param num_users: 客户端总数
        :return: group_matrix (num_users, num_groups), 0/1表示是否属于该组
        """
        group_matrix = np.zeros((num_users, self.num_groups), dtype=int)
        for i in range(num_users):
            # 每个客户端随机分配到overlap_ratio比例的分组
            n_groups = max(1, int(self.overlap_ratio * self.num_groups))
            groups = np.random.choice(self.num_groups, n_groups, replace=False)
            group_matrix[i, groups] = 1
        return group_matrix

    def group_aggregate(self,
                        global_params: Dict[str, torch.Tensor],
                        client_params_list: List[Dict[str, torch.Tensor]],
                        group_matrix: np.ndarray) -> List[Dict[str, torch.Tensor]]:
        """
        按分组聚合客户端参数
        :param global_params: 全局参数字典（用于初始化组参数）
        :param client_params_list: 各客户端参数字典列表
        :param group_matrix: 客户端-组分配矩阵
        :return: 各组的聚合参数字典列表
        """
        group_params_list = []
        for group_id in range(self.num_groups):
            # 初始化组参数为全局参数
            group_params = copy.deepcopy(global_params)

            # 收集属于该组的客户端参数
            group_clients = [i for i in range(len(client_params_list)) if group_matrix[i, group_id] == 1]
            if not group_clients:  # 空组处理
                group_params_list.append(group_params)
                continue

            # 对每个参数层计算FedAvg
            for key in group_params.keys():
                group_layer = torch.stack([client_params_list[i][key] for i in group_clients])
                group_params[key] = torch.mean(group_layer, dim=0)

            group_params_list.append(group_params)
        return group_params_list

    def compute_llr(self,
                    group_matrix: np.ndarray,
                    group_utilities: List[float]) -> np.ndarray:
        """
        计算每个客户端的后验似然比（LLR）
        :param group_matrix: 客户端-组分配矩阵
        :param group_utilities: 各组的效用值（如测试准确率）
        :return: llrs (num_users,) 每个客户端的LLR得分
        """
        llrs = np.zeros(group_matrix.shape[0])
        for client_id in range(group_matrix.shape[0]):
            # 累加客户端所属所有组的效用
            llr = sum(group_utilities[group_id] for group_id in np.where(group_matrix[client_id] == 1)[0])
            llrs[client_id] = llr
        return llrs

    def run_defense(self,
                    global_params: Dict[str, torch.Tensor],
                    client_params_list: List[Dict[str, torch.Tensor]],
                    data_loader: torch.utils.data.Dataset,
                    model: torch.nn.Module,
                    args ,
                    num_eliminate: int) -> list[dict[str, Tensor]]:
        """
        执行FedGT防御
        :param args:
        :param global_params: 当前全局参数字典
        :param client_params_list: 客户端参数字典列表
        :param data_loader: 验证数据加载器（用于计算组效用）
        :param model: 模型实例（用于加载参数进行测试）
        :param num_eliminate: 需剔除的恶意客户端数量
        :return: 被剔除的客户端索引列表
        """
        # 步骤1: 创建分组矩阵
        group_matrix = self.create_group_matrix(len(client_params_list))

        # 步骤2: 按组聚合参数
        group_params_list = self.group_aggregate(global_params, client_params_list, group_matrix)

        # 步骤3: 计算各组效用（测试准确率）
        group_utilities = []
        for group_params in group_params_list:
            # 加载组参数到模型
            model.load_state_dict(group_params)
            model.to(args.device)
            # 计算测试准确率（需实现test函数）
            accuracy,loss = test_img(model, data_loader, args)
            group_utilities.append(accuracy)

        # 步骤4: 计算LLR并剔除低分客户端
        llrs = self.compute_llr(group_matrix, group_utilities)
        eliminate_indices = np.argsort(llrs)[:num_eliminate].tolist()
        benign_params_list = [params for i, params in enumerate(client_params_list) if i not in eliminate_indices]
        return benign_params_list


class FedNorAvg:
    def __init__(self, dataset_name: str):
        self.norm_before_aggregation = self._get_norm_by_dataset(dataset_name)

    def _get_norm_by_dataset(self, dataset_name: str) -> float:
        """根据数据集名称返回基准范数"""
        norms = {
            "cifar": 15.0,
            "mnist": 16.5,
            "f-mnist": 20.0,
        }
        return norms.get(dataset_name, 5000.0)  # 默认值

    def aggregate(
        self,
        client_params_list: List[Dict[str, torch.Tensor]],
        global_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        聚合客户端参数，并做归一化，使最终范数接近设定标准
        :param client_params_list: 客户端参数字典列表
        :param global_params: 当前全局参数，用于结构模板
        :return: 聚合后的参数字典
        """
        # 平均聚合
        aggregated_params = {}
        for key in global_params:
            layer_stack = torch.stack([client[key].to(global_params[key].device) for client in client_params_list])
            aggregated_params[key] = torch.mean(layer_stack, dim=0)

        # 归一化使整体参数范数接近设定值
        vec = torch.cat([v.flatten() for v in aggregated_params.values()])
        current_norm = torch.norm(vec).item()
        scale_factor = self.norm_before_aggregation / (current_norm + 1e-10)

        for key in aggregated_params:
            aggregated_params[key] *= scale_factor

        return aggregated_params

class RFA:
    def __init__(self, max_iter: int = 10, eps: float = 1e-6, nu: float = 1e-6):
        self.max_iter = max_iter
        self.eps = eps
        self.nu = nu

    def _flatten_params(self, param_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([v.flatten() for v in param_dict.values()])

    def _unflatten_params(self, flat_tensor: torch.Tensor, template: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        new_params = {}
        idx = 0
        for k, v in template.items():
            numel = v.numel()
            new_params[k] = flat_tensor[idx:idx+numel].view_as(v).clone()
            idx += numel
        return new_params

    def aggregate(self, param_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        n = len(param_list)
        device = next(iter(param_list[0].values())).device
        weights = torch.ones(n, device=device) / n

        flat_params = [self._flatten_params(p).to(device) for p in param_list]
        stacked = torch.stack(flat_params)
        v = torch.mean(stacked, dim=0)  # 初始迭代点

        for _ in range(self.max_iter):
            distances = torch.norm(stacked - v, dim=1).clamp(min=self.nu)
            beta = weights / distances
            beta_sum = beta.sum()
            new_v = torch.sum((beta[:, None] * stacked), dim=0) / beta_sum
            if torch.norm(new_v - v) < self.eps:
                break
            v = new_v

        return self._unflatten_params(v, param_list[0])

def dnc(client_updates: List[Dict[str, torch.Tensor]], drop_ratio: float = 0.3, target_dim: int = 15) -> Dict[str, torch.Tensor]:
    """
    Divide-and-Conquer (DnC) 防御策略
    :param client_updates: 所有客户端上传的参数更新列表
    :param drop_ratio: 需要丢弃的最大投影客户端比例（如20%）
    :param target_dim: PCA 降维目标维度
    :return: 聚合后的参数字典
    """
    n = len(client_updates)
    flattened_updates = []
    for update in client_updates:
        flat = torch.cat([v.flatten() for v in update.values()]).cpu().numpy()
        flattened_updates.append(flat)
    flattened_updates = np.stack(flattened_updates)  # shape: [n_clients, d]

    # Step 1: PCA降维 + 计算主成分
    pca = PCA(n_components=target_dim)
    reduced = pca.fit_transform(flattened_updates)  # shape: [n_clients, target_dim]

    # Step 2: 取第一主成分方向
    principal_dir = pca.components_[0]  # shape: [d,]
    projections = reduced[:, 0]  # shape: [n_clients,]

    # Step 3: 根据投影值进行过滤
    num_drop = int(n * drop_ratio)
    drop_indices = np.argsort(np.abs(projections))[-num_drop:]  # 删除投影值最大的

    # Step 4: 保留其余客户端并进行平均
    selected_updates = [client_updates[i] for i in range(n) if i not in drop_indices]

    aggregated = copy.deepcopy(selected_updates[0])
    for k in aggregated.keys():
        for i in range(1, len(selected_updates)):
            aggregated[k] += selected_updates[i][k]
        aggregated[k] = aggregated[k] / len(selected_updates)
    return aggregated

class CenteredClipping:
    def __init__(self, tau: float = 100.0, num_iter: int = 1):
        """
        Centered Clipping 聚合器
        :param tau: clipping 半径，控制更新幅度
        :param num_iter: 内部迭代次数（论文中建议设为 1）
        """
        self.tau = tau
        self.num_iter = num_iter

    def _flatten_params(self, param_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([v.view(-1) for v in param_dict.values()])

    def _unflatten_params(self, flat_tensor: torch.Tensor, template: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """将扁平参数向量恢复为参数字典形式"""
        param_dict = {}
        pointer = 0
        for k, v in template.items():
            numel = v.numel()
            param_dict[k] = flat_tensor[pointer:pointer + numel].view_as(v)
            pointer += numel
        return param_dict

    def aggregate(self,
                  client_models: List[Dict[str, torch.Tensor]],
                  global_model: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        聚合完整的本地模型参数，输出聚合后的新模型
        """
        # Step 1: 计算所有客户端的 delta（Δw）
        deltas = []
        for client_model in client_models:
            delta = self._flatten_params({
                k: client_model[k] - global_model[k] for k in global_model
            })
            deltas.append(delta)

        # Step 2: Centered Clipping 聚合（论文中设定：初始点 v = 0 向量）
        v = torch.zeros_like(deltas[0])
        for _ in range(self.num_iter):
            updates = []
            for d in deltas:
                diff = d - v
                norm = torch.norm(diff) + 1e-6  # 避免除0
                clipped = diff * min(1.0, self.tau / norm)
                updates.append(clipped)
            v = v + sum(updates) / len(updates)

        # Step 3: 返回新的全局模型（w_global + Δ）
        update_dict = self._unflatten_params(v, global_model)
        new_global = copy.deepcopy(global_model)
        for k in new_global:
            new_global[k] += update_dict[k]

        return new_global

class LoMarDefense:
    def __init__(self, k: int = 10, threshold: float = 1.0, pca_dim: int = 20):
        """
        LoMar防御策略：局部密度估计驱动的客户端异常识别
        :param k: 每个客户端的邻居数
        :param threshold: 密度比阈值，小于该值视为异常
        :param pca_dim: PCA降维目标维度（不能超过n_clients - 1）
        """
        self.k = k
        self.threshold = threshold
        self.pca_dim = pca_dim

    def _flatten(self, model: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([v.detach().cpu().view(-1) for v in model.values()])

    def aggregate(self,
                  client_models: List[Dict[str, torch.Tensor]],
                  global_model: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Step 1: 计算每个客户端的更新 Δw_i
        deltas = [self._flatten(client) - self._flatten(global_model) for client in client_models]
        full_matrix = torch.stack(deltas).cpu().numpy()  # [n_clients, dim]

        # Step 2: 全局 PCA 降维
        n_clients = len(full_matrix)
        target_dim = min(self.pca_dim, n_clients - 1)
        pca = PCA(n_components=target_dim)
        update_matrix = pca.fit_transform(full_matrix)  # [n_clients, target_dim]

        # Step 3: 构建邻接图
        nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(update_matrix)
        distances, indices = nbrs.kneighbors(update_matrix)

        # Step 4: 计算每个客户端的密度比 F(i)
        F_scores = []
        for i in range(n_clients):
            neighbors_idx = indices[i][1:]  # 排除自己
            neighbors = update_matrix[neighbors_idx].T   # shape: [dim, k]
            target = update_matrix[i].reshape(-1, 1)      # shape: [dim, 1]

            dim, k_actual = neighbors.shape
            if dim > k_actual:
                local_pca = PCA(n_components=k_actual)
                neighbors = local_pca.fit_transform(neighbors.T).T     # shape: [k, dim] -> [k, k] -> [k, dim]
                target = local_pca.transform(target.T).T               # shape: [1, dim] -> [1, k] -> [k, 1]

            # KDE using sklearn (works even when covariance is singular)
            kde = KernelDensity(kernel='gaussian', bandwidth=1.0)
            kde.fit(neighbors.T)  # shape: [k, dim]

            q_i = np.exp(kde.score_samples(target.T))[0]
            q_neighbors = np.exp(kde.score_samples(neighbors.T)).mean()
            F_i = q_neighbors / (q_i + 1e-8)  # 避免除0
            F_scores.append(F_i)

        # Step 5: 基于阈值筛选可信客户端
        F_scores = np.array(F_scores)
        trusted_idx = np.where(F_scores >= self.threshold)[0]

        if len(trusted_idx) == 0:
            print("⚠️ [LoMar] 所有客户端都被视为异常，回退 FedAvg")
            trusted_idx = list(range(n_clients))

        trusted_deltas = [deltas[i] for i in trusted_idx]
        avg_update = sum(trusted_deltas) / len(trusted_deltas)

        # Step 6: 更新全局模型参数
        flat_global = self._flatten(global_model)
        new_flat = flat_global + avg_update

        # Step 7: 还原参数结构
        new_model = {}
        pointer = 0
        for k, v in global_model.items():
            numel = v.numel()
            new_model[k] = new_flat[pointer:pointer + numel].view_as(v)
            pointer += numel

        return new_model

class RoFL:
    def __init__(self, norm_type="L2", r=2.0, dynamic_bound=True, ema_beta=0.8):
        """
        :param norm_type: "L2" or "Linf"
        :param r: multiplier for robust median
        :param dynamic_bound: whether to use dynamic bound
        :param ema_beta: smoothing factor for bound history
        """
        self.norm_type = norm_type
        self.r = r
        self.dynamic_bound = dynamic_bound
        self.ema_beta = ema_beta
        self.smoothed_bound = None

    def _flatten(self, model: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([v.detach().cpu().view(-1) for v in model.values()])

    def _unflatten(self, flat: torch.Tensor, ref: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        new_model = {}
        ptr = 0
        for k, v in ref.items():
            numel = v.numel()
            new_model[k] = flat[ptr:ptr+numel].view_as(v)
            ptr += numel
        return new_model

    def _compute_norm(self, vec: np.ndarray) -> float:
        if self.norm_type == "L2":
            return np.linalg.norm(vec)
        elif self.norm_type == "Linf":
            return np.max(np.abs(vec))
        else:
            raise ValueError("Unsupported norm type")

    def _clip_soft(self, vec: np.ndarray, bound: float) -> np.ndarray:
        # Soft clipping: scale down large values smoothly
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

    def aggregate(self, updates: List[Dict[str, torch.Tensor]], ref_model: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], float]:
        flat_updates = [self._flatten(u).numpy() for u in updates]
        norms = [self._compute_norm(u) for u in flat_updates]

        # Robust median with EMA smoothing
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

        clipped = [self._clip_soft(u, bound) for u in flat_updates]
        avg_update_np = np.mean(clipped, axis=0)
        avg_update = torch.tensor(avg_update_np)
        return self._unflatten(avg_update, ref_model), bound
