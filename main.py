import copy

import torch
from models.attack import MaliciousParams
from utils.apply_attack_and_defense import apply_defense
from utils.getGrads import calculate_gradients_from_two_rounds, calculate_negative_gradients_from_two_rounds
from utils.participants import buildParticipants
from utils.queue import ParamQueue
from utils.saveWithCVS import save_accuracy_to_csv

from torchvision import datasets, transforms
from utils.sampling import partition_dataset
from utils.parameters import args_parser
from models.update import LocalUpdate
from models.nets import CNNFmnist, FCNetMNIST, AlexNet_CIFAR10
from models.test import test_img


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    #args.dataset = 'mnist'
    #args.dataset = 'cifar'
    #args.model = 'fc'
    #args.iid = True
    args.alpha=0.7 #0.3，0.5，0.7，1.0，IID
    #args.num_malicious=15
    defense_type = "Trimmed-mean"#FedAvg,Multi-Krum,Bulyan,CC,RFA,RoFL,Trimmed-mean:
    attack_type = "FedSDF"
    safe_min_ratio = 0.7 # 0.7
    safe_max_ratio = 1.6 # mnist：7.6，f-mnist：1.6，cifar10：1.86
    # cifar10：12-8，#f-mnist：12-15 ，mnist：30-20 for alpha=0.5
    center = 12
    scale = 15
    use_adaptive_norm = True
    STOP_ACC_THRESH = 10.0  # 测试精度达到该值则视为“崩溃”
    STOP_CONSEC_ROUNDS = 2  # 连续多少轮满足阈值才停止
    STOP_FILL_VALUE = 10.0  # 用于填充剩余轮次的值（通常与阈值相同）

    # load dataset and split users
    """
    定义了一系列数据转换操作并将其组合成一个转换管道，其中包括将图像转换为张量（transforms.ToTensor())和归一化操作(transforms.Normalize())
    transforms.Compose()是一个组合多个数据转换操作的函数,这里它将两个数据转换操作组合在一起形成一个转换管道.
    """
    if args.dataset == 'f-mnist':
        trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        #加载训练集和测试集
        dataset_train = datasets.FashionMNIST('../dataset/fashion_mnist/', train=True, download=True, transform=trans_fmnist)
        dataset_test = datasets.FashionMNIST('../dataset/fashion_mnist/', train=False, download=True, transform=trans_fmnist)
        # sample users
        #划分数据集
        dict_users = partition_dataset(dataset_train, args.num_users,args.iid,args.alpha)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        dataset_train = datasets.CIFAR10('../dataset/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../dataset/cifar', train=False, download=True, transform=trans_cifar)
        dict_users = partition_dataset(dataset_train, args.num_users,args.iid,args.alpha)
    elif args.dataset == 'mnist':
        trans_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST标准归一化参数
        ])
        dataset_train = datasets.MNIST(root='../dataset/mnist',train=True,download=True,transform=trans_mnist)
        dataset_test = datasets.MNIST(root='../dataset/mnist',train=False,download=True,transform=trans_mnist)
        dict_users = partition_dataset(dataset_train, args.num_users, args.iid, args.alpha)
    else:
        exit('Error: unrecognized dataset')
    #获取训练集中第一个样本的图像大小,并将其赋值给变量img_size
    img_size = dataset_train[0][0].shape

    # build model
    #根据参数选择模型类型和训练集类型

    if args.model == 'cnn' and args.dataset == 'f-mnist':
        net_glob = CNNFmnist().to(args.device)
    elif  args.dataset == 'cifar':
        net_glob = AlexNet_CIFAR10().to(args.device)
    elif args.model == 'fc' and args.dataset == 'mnist':
        net_glob = FCNetMNIST().to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    #启用模型的训练模式
    net_glob.train()
    """
    以下部分实现了联邦学习的训练过程.首先将全局模型的权重复制到每个客户端进行局部训练,然后根据一定策略聚合客户端的权重
    更新全局模型并打印每轮训练的平均loss值
    """
    # training
    #构建客户端（包括良性的和恶意的）
    participants = buildParticipants(args)
    acc_tests = []
    param_queue = ParamQueue()
    param_queue.add_params(copy.deepcopy(net_glob))
    state = {}
    state1 = {}
    attacker = MaliciousParams(safe_max_ratio,center,scale,safe_min_ratio,use_adaptive_norm = use_adaptive_norm)
    params_malicious = None

    #聚合过程
    for iters in range(args.epochs):
        w_locals = []
        w_malicious = []
        num_malicious=0
        for idx in range(args.num_users):
            if participants[idx].is_benign is True:
                #良性客户端执行本地更新（正常训练）
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                #传入当前的全局模型副本,并获取更新后的权重和局部损失
                w = local.train(net=copy.deepcopy(net_glob).to(args.device))
                #如果是所有客户端,将更新后的权重w赋值给w_locals的对应索引位置,否则，添加权重w到w_locals列表中
                w_locals.append(copy.deepcopy(w))
            elif iters > 0 and participants[idx].is_benign is False:
            #else:

                #params_malicious = apply_attack(attack_type, w_malicious, copy.deepcopy(net_glob.state_dict()))

                num_malicious+=1
                if num_malicious == 1:
                    params_malicious, state, state1 = attacker.malicious_update(copy.deepcopy(net_glob).state_dict(),
                                                                                calculate_gradients_from_two_rounds(
                                                                                    param_queue.get_params()[-2],
                                                                                    param_queue.get_params()[-1],
                                                                                    args.lr),
                                                                                calculate_negative_gradients_from_two_rounds(
                                                                                    param_queue.get_params()[-2],
                                                                                    param_queue.get_params()[-1],
                                                                                    args.lr),
                                                                                state, state1)




        if iters>0:
            for _ in range(args.num_malicious):
                w_locals.append(params_malicious)

        # update global weights
        print(len(w_locals))
        w_glob = apply_defense(defense_type,w_locals,copy.deepcopy(net_glob).state_dict(),params_malicious)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        param_queue.add_params(copy.deepcopy(net_glob))
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print('Round {:3d}'.format(iters))
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        acc_tests.append(acc_test)
        net_glob.train()
        # --- early stop: if test accuracy <= STOP_ACC_THRESH for STOP_CONSEC_ROUNDS consecutive rounds ---
        if len(acc_tests) >= STOP_CONSEC_ROUNDS:
            # 检查最近 STOP_CONSEC_ROUNDS 轮是否都 <= 阈值
            recent = acc_tests[-STOP_CONSEC_ROUNDS:]
            if all(a <= STOP_ACC_THRESH for a in recent):
                # 计算还需填充的轮数（使用 acc_tests 的当前长度）
                remaining = args.epochs - len(acc_tests)
                if remaining > 0:
                    acc_tests.extend([STOP_FILL_VALUE] * remaining)
                print(
                    f"[Early stop] detected {STOP_CONSEC_ROUNDS} consecutive rounds with test acc <= {STOP_ACC_THRESH}. "
                    f"Filling remaining {remaining} rounds with {STOP_FILL_VALUE} and stopping training.")
                break
        # --- end early stop ---

# save
    save_accuracy_to_csv(acc_tests, defense_type, attack_type, args.dataset)