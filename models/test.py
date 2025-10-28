"""
将最后得到的模型效果拿来测试
@:param1:要测试的模型
@:param2:测试数据集
@:param3:其他参数
"""
import torch.nn.functional as F
from torch.utils.data import DataLoader

def test_img(net_g,dataset,args):
    net_g.eval()
    #计算测试损失和正确分类的样本数
    test_loss = 0
    correct = 0
    data_loader = DataLoader(dataset,batch_size=args.bs)
    l = len(data_loader)
    #对数据加载器进行迭代,每次迭代获取一定批量的数据和对应的标签
    for idx,(data,target) in enumerate(data_loader):
        if args.gpu != -1:
            data,target = data.cuda(),target.cuda()
        #调用神经网络对数据向前传播
        log_probs = net_g(data)
        #使用交叉熵损失函数F.cross_entropy计算损失并累加到test_loss中
        test_loss += F.cross_entropy(log_probs,target,reduction='sum').item()
        #利用预测的对数概率计算预测的类别,并与目标标签比较,统计正确分类的样本数
        y_pred = log_probs.data.max(1,keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    #计算平均测试损失和准确率
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss