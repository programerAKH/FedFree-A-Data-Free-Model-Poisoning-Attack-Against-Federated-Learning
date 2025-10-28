import torch
import copy

from models.attack_compare import PoisonedFLAttacker   # ✅ 使用 PoisonedFL
from models.nets import FCNetMNIST, CNNFmnist, AlexNet_CIFAR10
from models.update import LocalUpdate
from models.test import test_img
from utils.parameters import args_parser
from utils.sampling import partition_dataset
from torchvision import datasets, transforms

from utils.apply_attack_and_defense import apply_defense_grad
from utils.saveWithCVS import save_accuracy_to_csv

# ---------------------- 参数配置 ---------------------- #
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

# 你也可以在命令行改 --dataset/--model/--iid 等
defense_type = "CC"   # 可选：FedAvg, Multi-Krum, Bulyan, CC, RFA, RoFL（与你工具函数一致）
attack_type  = "PoisonedFL"
args.dataset = 'cifar'#cifar，mnist
#args.model = 'fc'

# ---------------------- 数据集 ---------------------- #
if args.dataset == 'f-mnist':
    trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset_train = datasets.FashionMNIST('../dataset/fashion_mnist/', train=True, download=True, transform=trans_fmnist)
    dataset_test  = datasets.FashionMNIST('../dataset/fashion_mnist/', train=False,  download=True, transform=trans_fmnist)
    dict_users = partition_dataset(dataset_train, args.num_users, args.iid, args.alpha)
elif args.dataset == 'cifar':
    trans_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    dataset_train = datasets.CIFAR10('../dataset/cifar', train=True, download=True, transform=trans_cifar)
    dataset_test  = datasets.CIFAR10('../dataset/cifar', train=False,  download=True, transform=trans_cifar)
    dict_users = partition_dataset(dataset_train, args.num_users, args.iid, args.alpha)
elif args.dataset == 'mnist':
    trans_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset_train = datasets.MNIST('../dataset/mnist', train=True, download=True, transform=trans_mnist)
    dataset_test  = datasets.MNIST('../dataset/mnist', train=False,  download=True, transform=trans_mnist)
    dict_users = partition_dataset(dataset_train, args.num_users, args.iid, args.alpha)
else:
    raise SystemExit('Error: unrecognized dataset')

# ---------------------- 初始化模型 ---------------------- #
if args.model == 'cnn' and args.dataset == 'f-mnist':
    net_glob = CNNFmnist().to(args.device)
elif args.dataset == 'cifar':
    net_glob = AlexNet_CIFAR10().to(args.device)
elif args.model == 'fc' and args.dataset == 'mnist':
    net_glob = FCNetMNIST().to(args.device)
else:
    raise SystemExit('Error: unrecognized model')

print(net_glob)

# ---------------------- 初始化 PoisonedFL 攻击器 ---------------------- #
attacker = PoisonedFLAttacker(
    num_malicious=args.num_malicious,
    device=args.device,
    as_gradient=True,      # 你的服务器是 w -= lr * grad 就设 True
    c0=20.0, e=25, rho=0.9, c_min=18.0, align_p=0.95,
    layered_v=True, warmup_rounds=5,
    beta0=0.80, eps0=1e-2  # 与防御无关的通用起点
)

# 把初始全局模型喂给攻击器（用于形成 w 历史；PoisonedFL 需要至少两轮 w 才能开始攻击）
attacker.add_global(copy.deepcopy(net_glob.state_dict()))

acc_list = []

# ---------------------- 联邦训练主循环 ---------------------- #
for epoch in range(args.epochs):
    print(f"\n--- Global Round {epoch+1} ---")
    net_glob.train()

    local_grads = []
    # 这里简单指定前 num_malicious 个客户端为恶意；你的框架若有随机抽样/选择逻辑，可在此替换
    malicious_idxs = list(range(args.num_malicious))

    # PoisonedFL 在有两份全局模型历史（w_{t-1}, w_{t-2}）时开始生成恶意更新
    can_attack = (len(attacker.w_hist) >= 2)
    mal_updates = attacker.generate_malicious_gradients(copy.deepcopy(net_glob.state_dict())) if can_attack else None
    m_ptr = 0
    for idx in range(args.num_users):
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])

        if idx in malicious_idxs and can_attack:
            grad = mal_updates[m_ptr % len(mal_updates)]
            m_ptr += 1
        else:
            grad = local.train_and_return_gradients(copy.deepcopy(net_glob))
        local_grads.append(grad)

    # -------------- 防御 + 聚合 -------------- #
    print(len(local_grads))
    avg_grad = apply_defense_grad(defense_type, local_grads)
    for k in avg_grad:
        avg_grad[k] = avg_grad[k].to(args.device)

    # -------------- 全局模型更新 -------------- #
    global_state = net_glob.state_dict()
    for k in global_state:
        global_state[k] -= args.lr * avg_grad[k]
    net_glob.load_state_dict(global_state)

    # 把更新后的全局模型再次喂给攻击器，形成连续 w 序列
    attacker.add_global(copy.deepcopy(net_glob.state_dict()))

    # -------------- 测试并记录 -------------- #
    acc_test, _ = test_img(net_glob, dataset_test, args)
    acc_list.append(acc_test)
    print(f"Test Accuracy at Epoch {epoch+1}: {acc_test:.2f}%")

# ---------------------- 保存曲线 ---------------------- #
save_accuracy_to_csv(acc_list, defense_type, attack_type, args.dataset)
print("Saved accuracy CSV.")