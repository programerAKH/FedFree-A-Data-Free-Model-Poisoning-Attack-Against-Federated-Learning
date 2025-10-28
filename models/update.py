"""
定义客户端本地更新的内容,包括参数和梯度两种方法
"""
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset



#这里定义了一个DatasetSplit类,继承自Dataset,该类能从原始数据集中创建一个仅包含特定样本的子数据集
class DatasetSplit(Dataset):
    def __init__(self,dataset,idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image,label = self.dataset[self.idxs[item]]
        return  image,label

#下面这个类用于本地客户端的模型训练和更新,在train方法中，
# 通过迭代数据加载器的批次，对模型进行前向传播、计算损失、反向传播和参数更新，最终返回模型的状态字典和训练周期的平均损失
class LocalUpdate(object):
    def __init__(self,args,dataset=None,idxs=None):
        #保存传入的参数,用于配置训练过程中的超参数
        self.args = args
        #保存一个交叉熵损失函数的实例,用于计算训练过程中的损失
        self.loss_func = nn.CrossEntropyLoss()
        #用于保存选择的客户端
        self.selected_clients = []
        #创建一个数据集加载器,加载一个子数据集,并随机打乱
        self.ldr_train = DataLoader(DatasetSplit(dataset,idxs),batch_size=self.args.local_bs,shuffle=True)

    def train(self,net):
        #将模型设置为训练模式
        net.train()
        #创建一个SGD优化器,使用net.parameters()作为参数,设置学习率和动量
        optimizer = torch.optim.SGD(net.parameters(),lr=self.args.lr,momentum=self.args.momentum)
        #用于保存每个训练周期的损失
        for iter in range(self.args.local_ep):
           for batch_idx,(images,labels) in enumerate(self.ldr_train):
               images,labels = images.to(self.args.device),labels.to(self.args.device)
               #清零模型参数的梯度
               net.zero_grad()
               #通过模型进行向前传播,获取预测的对数概率
               log_probs = net(images)
               #使用损失函数计算损失
               loss = self.loss_func(log_probs,labels)
               #对损失反向传播并更新参数
               loss.backward()
               optimizer.step()
               #打印某些批次的训练进度和损失

        #返回模型的状态字典和所有周期的平均损失
        return net.state_dict()

    def train_and_return_gradients(self, net):
        net = copy.deepcopy(net).to(self.args.device)
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9)
        loss_fn = torch.nn.CrossEntropyLoss()

        # 保存初始参数 θ_init
        theta_init = copy.deepcopy(net.state_dict())

        for epoch in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

        # 获取训练后参数 θ_final
        theta_final = net.state_dict()
        grads = {}
        for k in theta_final:
            grads[k] = (theta_init[k] - theta_final[k]) / self.args.lr

        return grads


