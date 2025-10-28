import random

class Participant:
    def __init__(self,is_benign=True):
        self.is_benign = is_benign

#构建客户端（包括35个良性客户端和15个恶意客户端）
def buildParticipants(args):
    malicious_idxs = random.sample(range(args.num_users), args.num_malicious)
    participants = []
    for i in range(args.num_users):
        if i not in malicious_idxs:
            participants.append(Participant())
        else:
            participants.append(Participant(is_benign=False))

    return participants