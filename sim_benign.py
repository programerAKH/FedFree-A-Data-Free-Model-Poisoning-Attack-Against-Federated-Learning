
import os
import copy
import csv
import torch
from models.shampoo import data_less_local_train
from utils.apply_attack_and_defense import apply_defense
from utils.getGrads import calculate_gradients_from_two_rounds
from utils.participants import buildParticipants
from utils.queue import ParamQueue
from torchvision import datasets, transforms
from utils.sampling import partition_dataset
from utils.parameters import args_parser
from models.update import LocalUpdate
from models.nets import CNNFmnist, FCNetMNIST, AlexNet_CIFAR10, ResNet34_CIFAR
from models.test import test_img


def flatten_params(params_dict):
    """将 state_dict 展平为向量"""
    return torch.cat([p.view(-1).float().cpu() for p in params_dict.values()])


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    defense_type = "FedAvg"

    # ================= 数据加载 =================
    if args.dataset == 'f-mnist':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('../dataset/fashion_mnist/', train=True, download=True, transform=trans)
        dataset_test = datasets.FashionMNIST('../dataset/fashion_mnist/', train=False, download=True, transform=trans)
    elif args.dataset == 'cifar':
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                         (0.2023, 0.1994, 0.2010))])
        dataset_train = datasets.CIFAR10('../dataset/cifar', train=True, download=True, transform=trans)
        dataset_test = datasets.CIFAR10('../dataset/cifar', train=False, download=True, transform=trans)
    elif args.dataset == 'mnist':
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../dataset/mnist', train=True, download=True, transform=trans)
        dataset_test = datasets.MNIST('../dataset/mnist', train=False, download=True, transform=trans)
    elif args.dataset == 'cifar100':
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                         (0.2675, 0.2565, 0.2761))])
        dataset_train = datasets.CIFAR100('../dataset/cifar100', train=True, download=True, transform=trans)
        dataset_test = datasets.CIFAR100('../dataset/cifar100', train=False, download=True, transform=trans)
    else:
        exit('Error: unrecognized dataset')

    dict_users = partition_dataset(dataset_train, args.num_users, args.iid, args.alpha)

    # ================= 模型构建 =================
    if args.dataset == 'f-mnist':
        net_glob = CNNFmnist().to(args.device)
    elif args.dataset == 'cifar':
        net_glob = AlexNet_CIFAR10().to(args.device)
    elif args.dataset == 'mnist':
        net_glob = FCNetMNIST().to(args.device)
    elif args.dataset == 'cifar100':
        net_glob = ResNet34_CIFAR(num_classes=100).to(args.device)
    else:
        exit('Error: unrecognized model')

    print(net_glob)
    net_glob.train()

    # ================= 训练准备 =================
    participants = buildParticipants(args)
    param_queue = ParamQueue()
    param_queue.add_params(copy.deepcopy(net_glob))
    state = {}
    params_sim = None

    # 保存结果文件夹
    save_dir = os.path.join("save", "shampoo", args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    metrics_file = os.path.join(
        save_dir,
        f"{args.dataset}_alpha{args.alpha}_mal{args.num_malicious}.csv"
    )

    metrics_log = []

    # ================= 训练循环 =================
    for iters in range(args.epochs):
        w_locals, benign_updates, malicious_updates = [], [], []
        num_malicious = 0

        for idx in range(args.num_users):
            if participants[idx].is_benign:
                # 良性客户端本地更新
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w = local.train(net=copy.deepcopy(net_glob).to(args.device))
                w_locals.append(copy.deepcopy(w))
                benign_updates.append(copy.deepcopy(w))
            elif iters > 0 and not participants[idx].is_benign:
                num_malicious += 1
                if num_malicious == 1:
                    params_sim, state = data_less_local_train(
                        copy.deepcopy(net_glob).state_dict(),
                        calculate_gradients_from_two_rounds(
                            param_queue.get_params()[-2],
                            param_queue.get_params()[-1],
                            args.lr),
                        state, t=5)

        # 所有恶意客户端提交相同的 OnlySimBenign 参数
        if iters > 0 and params_sim is not None:
            for _ in range(args.num_malicious):
                w_locals.append(copy.deepcopy(params_sim))
                malicious_updates.append(copy.deepcopy(params_sim))

        # FedAvg 更新
        w_glob = apply_defense(defense_type, w_locals,
                               copy.deepcopy(net_glob).state_dict(), params_sim)
        net_glob.load_state_dict(w_glob)
        param_queue.add_params(copy.deepcopy(net_glob))

        # 测试
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)

        # ===== 指标计算 =====
        if benign_updates and malicious_updates:
            # 平均良性
            benign_mean = copy.deepcopy(benign_updates[0])
            for k in benign_mean:
                benign_mean[k] = sum(b[k] for b in benign_updates) / len(benign_updates)
            # 平均恶意
            mal_mean = copy.deepcopy(malicious_updates[0])
            for k in mal_mean:
                mal_mean[k] = sum(m[k] for m in malicious_updates) / len(malicious_updates)

            benign_vec = flatten_params(benign_mean)
            mal_vec = flatten_params(mal_mean)

            cosine = torch.nn.functional.cosine_similarity(
                benign_vec, mal_vec, dim=0).item()
            norm_ratio = mal_vec.norm().item() / (benign_vec.norm().item() + 1e-10)
        else:
            cosine, norm_ratio = None, None

        metrics_log.append([iters, acc_test, cosine, norm_ratio])
        print(f"Round {iters}: Acc={acc_test:.4f}, Cos={cosine}, NormRatio={norm_ratio}")

        net_glob.train()

    # ================= 保存到 CSV =================
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "acc_test", "cosine", "norm_ratio"])
        writer.writerows(metrics_log)

    print(f"Metrics saved to {metrics_file}")
