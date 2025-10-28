import copy

from models.agg_grads import FedAvg_grads, multi_krum_grads, bulyan_grads, CenteredClipping_Grad, RFA_Grad, RoFL_Grad, \
    trimmed_mean_grads
from models.aggregation import FedAvg, multi_krum, bulyan, CenteredClipping, RFA, RoFL, trimmed_mean
from models.attack_compare import LittleIsEnoughAttack, MinMaxAttack, MinSumAttack, FangAttack,generate_malicious_model, ExponentialSmoothing
from models.nets import CNNFmnist, AlexNet_CIFAR10, FCNetMNIST


def apply_attack(
    attack_type: str,
    client_state_dicts,
    global_model_state_dict=None,
    net_glob=None,
    args=None,
    dataset_train=None,
    dict_users=None,
    participants=None,
):
    """
    根据攻击类型生成恶意更新。
    仅 FMPA 类型维护历史 global model state_dict，用于预测参考模型。
    """

    if attack_type == "LIE":
        attacker = LittleIsEnoughAttack(z_max=1.0)
        return attacker.generate_poisoned_update(client_state_dicts)

    elif attack_type == "MinMax":
        attacker = MinMaxAttack()
        return attacker.generate_poisoned_update(client_state_dicts)

    elif attack_type == "MinSum":
        attacker = MinSumAttack()
        return attacker.generate_poisoned_update(client_state_dicts)

    elif attack_type == "Fang":
        if global_model_state_dict is None:
            raise ValueError("Fang attack requires the global model state_dict.")
        attacker = FangAttack()
        return attacker.generate_poisoned_models(global_model_state_dict, client_state_dicts, 50, 15)

    elif attack_type == "I-FMPA":
        from models.attack_compare import generate_malicious_model, ExponentialSmoothing
        from utils.build_malicious_list import build_malicious_dataloaders
        from models.nets import FCNetMNIST, CNNFmnist, AlexNet_CIFAR10

        # 初始化历史状态追踪
        if not hasattr(apply_attack, "_historical_fmpa_states"):
            apply_attack._historical_fmpa_states = []
        apply_attack._historical_fmpa_states.append(copy.deepcopy(net_glob.state_dict()))
        if len(apply_attack._historical_fmpa_states) > 10:
            apply_attack._historical_fmpa_states.pop(0)

        # 构造参考模型（预测的下一轮 global model）
        predictor = ExponentialSmoothing(alpha=0.7)
        predictor.update(apply_attack._historical_fmpa_states[-1])
        reference_model = predictor.predict()

        # 构建模型工厂函数（适配参数）
        def model_fn():
            if args.model == 'cnn' and args.dataset == 'f-mnist':
                return CNNFmnist()
            elif args.dataset == 'cifar':
                return AlexNet_CIFAR10()
            elif args.model == 'fc' and args.dataset == 'mnist':
                return FCNetMNIST()
            else:
                raise ValueError("Unsupported model/dataset combination")

        # 构建恶意客户端数据
        malicious_data = build_malicious_dataloaders(dataset_train, dict_users, participants, args, val_ratio=0.2)

        # 调用 I-FMPA 恶意参数生成函数
        params_malicious = generate_malicious_model(
            reference_model=reference_model,
            model_fn=model_fn,
            malicious_clients_data_list=malicious_data,
            threshold=1.0 / args.num_classes,
            lambda_=1.0,
            device=args.device,
            historical_updates=apply_attack._historical_fmpa_states
        )

        return params_malicious

    else:
        raise ValueError(f"Unknown attack: {attack_type}")


def apply_defense(defense_type: str, client_state_dicts,model_dict,param_mal=None):
    if defense_type == "FedAvg":
        glob_dict = FedAvg(client_state_dicts)

    elif defense_type == "Multi-Krum":
        glob_dict,_ = multi_krum(client_state_dicts, 15, 35,param_mal)
        print(_)

    elif defense_type == "Bulyan":
        glob_dict,_ = bulyan(client_state_dicts, 15, 35, 10,param_mal)
        print(_)
    elif defense_type == "CC":
        defender = CenteredClipping()
        glob_dict = defender.aggregate(client_state_dicts, model_dict)

    elif defense_type == "RFA":
        defender = RFA()
        glob_dict = defender.aggregate(client_state_dicts)

    elif defense_type == "RoFL":
        defender = RoFL()
        glob_dict,_ =defender.aggregate(client_state_dicts, model_dict)

    elif defense_type == "Trimmed-mean":
        glob_dict = trimmed_mean(client_state_dicts,15)

    else:
        raise ValueError(f"Unknown defense: {defense_type}")

    return glob_dict

def apply_defense_grad(defense_type: str, client_grads):
    """
    梯度版防御策略聚合器
    :param defense_type: 防御方法名称（FedAvg, Multi-Krum, Bulyan, CC, RFA, RoFL）
    :param client_grads: 客户端上传的梯度列表（List[OrderedDict]）
    :return: 聚合后的梯度字典
    """
    if defense_type == "FedAvg":
        glob_grad = FedAvg_grads(client_grads)

    elif defense_type == "Multi-Krum":
        glob_grad = multi_krum_grads(client_grads, f=15, m=35)  # 你可根据实验调整

    elif defense_type == "Bulyan":
        glob_grad = bulyan_grads(client_grads, f=15, m=35, beta=10)

    elif defense_type == "CC":
        defender = CenteredClipping_Grad()
        glob_grad = defender.aggregate(client_grads)

    elif defense_type == "RFA":
        defender = RFA_Grad()
        glob_grad = defender.aggregate(client_grads)

    elif defense_type == "RoFL":
        defender = RoFL_Grad()
        glob_grad, bound = defender.aggregate(client_grads)
        print(f"[RoFL] dynamic bound: {bound:.4f}")

    elif defense_type == "Trimmed-mean":
        glob_grad = trimmed_mean_grads(client_grads,15)

    else:
        raise ValueError(f"Unknown defense: {defense_type}")

    return glob_grad
