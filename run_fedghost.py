import torch
import copy

from models.agg_grads import FedAvg_grads
from models.attack_compare import FedGhostAttacker
from models.nets import FCNetMNIST, CNNFmnist, AlexNet_CIFAR10
from models.update import LocalUpdate
from models.test import test_img
from utils.parameters import args_parser
from utils.sampling import partition_dataset
from torchvision import datasets, transforms

from utils.apply_attack_and_defense import apply_defense, apply_defense_grad
from utils.saveWithCVS import save_accuracy_to_csv  # ✅ 导入你的保存工具

# ---------------------- 参数配置 ---------------------- #
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
#args.dataset = 'mnist'
#args.dataset = 'cifar'
#args.model = 'fc'
#args.iid = True
defense_type = "RoFL"#FedAvg,Multi-Krum,Bulyan,CC,RFA,RoFL
attack_type = "FedGhost"
if args.dataset == 'f-mnist':
    trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # 加载训练集和测试集
    dataset_train = datasets.FashionMNIST('../dataset/fashion_mnist/', train=True, download=True,
                                          transform=trans_fmnist)
    dataset_test = datasets.FashionMNIST('../dataset/fashion_mnist/', train=False, download=True,
                                         transform=trans_fmnist)
    # sample users
    # 划分数据集
    dict_users = partition_dataset(dataset_train, args.num_users, args.iid, args.alpha)
elif args.dataset == 'cifar':
    trans_cifar = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    dataset_train = datasets.CIFAR10('../dataset/cifar', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR10('../dataset/cifar', train=False, download=True, transform=trans_cifar)
    dict_users = partition_dataset(dataset_train, args.num_users, args.iid, args.alpha)
elif args.dataset == 'mnist':
    trans_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST标准归一化参数
    ])
    dataset_train = datasets.MNIST(root='../dataset/mnist', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST(root='../dataset/mnist', train=False, download=True, transform=trans_mnist)
    dict_users = partition_dataset(dataset_train, args.num_users, args.iid, args.alpha)
else:
    exit('Error: unrecognized dataset')
# ---------------------- 初始化模型 ---------------------- #
if args.model == 'cnn' and args.dataset == 'f-mnist':
    net_glob = CNNFmnist().to(args.device)
elif  args.dataset == 'cifar':
    net_glob = AlexNet_CIFAR10().to(args.device)
elif args.model == 'fc' and args.dataset == 'mnist':
    net_glob = FCNetMNIST().to(args.device)
else:
    exit('Error: unrecognized model')
print(net_glob)

# ---------------------- 初始化攻击器（论文版） ---------------------- #
# === MOD: 使用改造后的 FedGhostAttacker（支持 ΔW/ΔG、多割线预测、concealed/sacrificial、反馈 γ）===
attacker = FedGhostAttacker(
    model_fn=lambda: copy.deepcopy(net_glob),
    num_malicious=args.num_malicious,
    window_size=10,           # 论文里常用 10 左右的窗口
    eta=args.lr,
    device=args.device,
    gamma_min=2.2,
    gamma_max=5.0,
    cmin=0.5,                 # 余弦阈值（>=则奖励放大γ）
    sacrificial_scale=10.0,   # 反向牺牲者的放大系数
    topk_ratio=1.0            # 如需只在Top-K幅值坐标注入，可改为0.2等
)

# === MOD: 先把初始全局模型喂给攻击器（用于形成 w 历史；g不可得时内部会用Δw/η近似）===
attacker.add_global(copy.deepcopy(net_glob.state_dict()), global_grad_state_dict=None)

acc_list = []

for epoch in range(args.epochs):
    print(f"\n--- Global Round {epoch+1} ---")
    net_glob.train()

    local_grads = []
    malicious_idxs = list(range(args.num_malicious))  # 前 num_malicious 个客户端为恶意

    # === MOD: 攻击触发条件：需要至少 3 个 (w,g) 历史点用于 ΔW/ΔG 预测 ===
    can_attack = (len(attacker.w_hist) >= 3) and (len(attacker.g_hist) >= 3)

    for idx in range(args.num_users):
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])

        if idx in malicious_idxs and can_attack:
            # === MOD: 使用“论文版”FedGhost 构造恶意梯度（concealed + sacrificial）===
            malicious_grads = attacker.generate_malicious_gradients(copy.deepcopy(net_glob.state_dict()),
                                                                    sacrifice_ratio=0.2)
            grad = malicious_grads[idx % len(malicious_grads)]
        else:
            # 正常 benign 客户端上传梯度
            grad = local.train_and_return_gradients(copy.deepcopy(net_glob))

        local_grads.append(grad)

    # -------------- 聚合与全局更新 -------------- #
    avg_grad = apply_defense_grad(defense_type, local_grads)
    for k in avg_grad:
        avg_grad[k] = avg_grad[k].to(args.device)

    # === MOD: 记录“攻击后的全局梯度”以做反馈自适应 γ（cs >= cmin → 放大γ，否则缩小）===
    attacker.add_after_attack_global_grad(copy.deepcopy(avg_grad))

    # 全局模型参数更新
    global_state = net_glob.state_dict()
    for k in global_state:
        global_state[k] -= args.lr * avg_grad[k]
    net_glob.load_state_dict(global_state)

    # === MOD: 把本轮更新后的全局模型再喂入，形成连续的 w 序列；
    #           若没有显式 g_t，这一步会用 Δw/η 自动近似填充 g_hist，满足 ΔW/ΔG 预测所需 ===
    attacker.add_global(copy.deepcopy(net_glob.state_dict()), global_grad_state_dict=None)

    # 测试准确率
    acc_test, _ = test_img(net_glob, dataset_test, args)
    acc_list.append(acc_test)
    print(f"Test Accuracy at Epoch {epoch+1}: {acc_test:.2f}%")

save_accuracy_to_csv(acc_list, defense_type, attack_type, args.dataset)