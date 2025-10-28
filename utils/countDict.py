import torch


def count_malicious_dicts(param_malicious, selected_updates):
    selected_malicious = 0

    for selected_dict in selected_updates:
        # 确保两个字典的键完全一致
        if selected_dict.keys() != param_malicious.keys():
            continue

        all_equal = True
        for key in param_malicious.keys():
            val1 = param_malicious[key]
            val2 = selected_dict[key]

            # 如果是张量，使用 torch.equal 或逐元素比较
            if isinstance(val1, torch.Tensor):
                if not torch.equal(val1, val2):
                    all_equal = False
                    break
            else:
                if val1 != val2:
                    all_equal = False
                    break

        if all_equal:
            selected_malicious += 1

    return selected_malicious