"""
划分数据集
"""
import numpy as np

def partition_dataset(dataset, num_users, iid=True, alpha=None):
    """
    Partition the dataset into `num_users` clients.

    Args:
        dataset: The dataset to be partitioned.
        num_users: Number of clients.
        iid: If True, the data is IID partitioned; otherwise, Non-IID.
        alpha: Dirichlet distribution parameter for Non-IID partitioning.
               Smaller alpha means more heterogeneous data distribution.

    Returns:
        dict_users: A dictionary where keys are user IDs and values are sets of data indices.
    """
    num_items = int(len(dataset) / num_users)
    dict_users = {i: set() for i in range(num_users)}
    all_idxs = [i for i in range(len(dataset))]

    if iid:
        # IID partitioning: randomly assign data to users
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
    else:
        # Non-IID partitioning: use Dirichlet distribution to assign data
        if alpha is None:
            raise ValueError("Alpha must be specified for Non-IID partitioning.")

        # Get labels from the dataset
        labels = np.array(dataset.targets)
        num_classes = len(np.unique(labels))

        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_users), num_classes)

        # Assign data to users based on the sampled proportions
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions_k = proportions[k]
            proportions_k = np.cumsum(proportions_k) * len(idx_k)
            proportions_k = (proportions_k).astype(int)

            for i in range(num_users):
                start = proportions_k[i]
                end = proportions_k[i + 1] if i < num_users - 1 else len(idx_k)
                dict_users[i].update(idx_k[start:end])

    return dict_users