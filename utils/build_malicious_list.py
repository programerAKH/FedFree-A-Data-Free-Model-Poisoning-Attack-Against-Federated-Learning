from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def build_malicious_dataloaders(dataset, dict_users, participants, args, val_ratio=0.2):
    """
    构建所有恶意客户端的 (train_loader, val_loader) 对，供 I-FMPA 使用。

    参数：
        dataset: 原始完整数据集（如 MNIST/CIFAR）
        dict_users: 客户端 id -> 样本索引 的映射
        participants: 每个客户端的 Participant 实例，包含 is_benign 属性
        args: 包含 local_bs, device 等属性的配置对象
        val_ratio: 用于划分验证集的比例，默认 20%

    返回：
        malicious_loaders: List of (train_loader, val_loader) 对
    """
    malicious_loaders = []

    for user_id, participant in enumerate(participants):
        if not participant.is_benign:
            idxs = list(dict_users[user_id])
            train_idxs, val_idxs = train_test_split(idxs, test_size=val_ratio, random_state=42)

            train_subset = Subset(dataset, train_idxs)
            val_subset = Subset(dataset, val_idxs)

            train_loader = DataLoader(train_subset, batch_size=args.local_bs, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=args.local_bs, shuffle=False)

            malicious_loaders.append((train_loader, val_loader))

    return malicious_loaders
