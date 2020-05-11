

import torch.optim as optim

def get_optimizer(optimizer_name, net_params, lr, eps):
    if optimizer_name == "RMSProp" or optimizer_name == "RMSprop":
        return optim.RMSprop(net_params, lr=lr, eps=eps)
    else:
        exit(0)