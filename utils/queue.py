from collections import deque

from torch import nn


class ParamQueue:
    def __init__(self, maxlen=5):
        """
        初始化参数队列，最大长度为2，存储两轮的参数
        maxlen: 队列最大长度，默认为2
        """
        self.queue = deque(maxlen=maxlen)  # 使用 deque 存储模型参数

    def add_params(self, model: nn.Module):
        """
        将当前模型参数加入队列
        """
        # 获取当前模型的参数字典并克隆，以防止修改原始参数
        current_params = {name: param.clone() for name, param in model.named_parameters()}
        self.queue.append(current_params)

    def get_params(self):
        """
        获取队列中的所有模型参数
        返回一个列表，包含队列中的参数字典
        """
        return list(self.queue)