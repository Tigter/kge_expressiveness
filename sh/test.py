import torch


def caculate_constarin(gamma_m, beta, alpha):
    pi = 3.14159265358979323846
    a0 = torch.tensor([0.0001,0.01,0.00000001])
    a_pi = torch.tensor([0.0001,0.01,0.00000001])

    penalty = torch.zeros_like(a0)
    penalty[a0 < gamma_m] = beta
    print(penalty)
    l_1 = (a0 * penalty).norm(p=2)
    
    penalty = torch.zeros_like(a_pi)
    penalty[a_pi < gamma_m] = beta
    l_2 = (a_pi * penalty).norm(p=2)
    print(l_2)
    loss = (l_1 + l_2) * alpha
    return loss


caculate_constarin(0.0001,1.5,0.25)

