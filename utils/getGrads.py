import torch
from typing import Dict



def calculate_gradients_from_two_rounds(initial_params: Dict[str, torch.Tensor],
                                        final_params: Dict[str, torch.Tensor],
                                        learning_rate: float) -> Dict[str, torch.Tensor]:
    """
    根据两轮训练的参数计算梯度（即参数的变化）。

    initial_params: 第一次训练后的参数字典
    final_params: 第二次训练后的参数字典
    learning_rate: 学习率
    返回: 计算出的梯度字典
    """
    gradients = {}

    # 遍历每个参数，计算梯度
    for name, initial_param in initial_params.items():
        if name in final_params:
            # 计算每个参数的变化，梯度近似为参数变化除以学习率
            final_param = final_params[name]
            gradient = (final_param - initial_param) / learning_rate
            gradients[name] = gradient.detach().clone()  # 克隆梯度，防止原地修改

    return gradients

def calculate_negative_gradients_from_two_rounds(initial_params: Dict[str, torch.Tensor],
                                        final_params: Dict[str, torch.Tensor],
                                        learning_rate: float) -> Dict[str, torch.Tensor]:
    """
    根据两轮训练的参数计算梯度（即参数的变化）。

    initial_params: 第一次训练后的参数字典
    final_params: 第二次训练后的参数字典
    learning_rate: 学习率
    返回: 计算出的梯度字典
    """
    gradients = {}

    # 遍历每个参数，计算梯度
    for name, initial_param in initial_params.items():
        if name in final_params:
            # 计算每个参数的变化，梯度近似为参数变化除以学习率
            final_param = final_params[name]
            gradient = -(final_param - initial_param) / learning_rate
            gradients[name] = gradient.detach().clone()  # 克隆梯度，防止原地修改

    return gradients