import torch
from torch import Tensor

def calculate_width(X: Tensor) -> Tensor:
    n = X.shape[0]
    G = torch.sum(X * X, dim=-1, keepdim=True)
    Q = G.repeat(1, n)
    R = G.permute(1, 0).repeat(n, 1)
    dists = Q + R - 2 * (X @ X.permute(1, 0))
    dists = dists - torch.tril(dists)
    dists = dists.view(n*n, 1)
    dists = dists[dists>0]
    if max(dists.shape) == 0:
        width_x = 0.0001
    else:
        width_x = torch.sqrt(0.5 * torch.median(dists))
    return width_x

def rbf_mul(pattern_1: Tensor, pattern_2: Tensor, deg: Tensor) -> Tensor:
    n = pattern_1.shape[0]
    G = torch.sum(pattern_1 * pattern_1, dim=-1, keepdim=True)
    H = torch.sum(pattern_2 * pattern_2, dim=-1, keepdim=True)
    Q = G.repeat(1, n)
    R = H.permute(1, 0).repeat(n, 1)
    H = Q + R - 2 * (pattern_1 @ pattern_2.permute(1, 0))
    H = torch.exp(-H / 2 / (deg**2))
    return H

def HSIC(X: Tensor, Y: Tensor) -> Tensor:
    width_x = calculate_width(X)
    width_y = calculate_width(Y)
    n = X.shape[0]
    H = torch.eye(n, device='cuda') - torch.ones((n, n), device='cuda') / n
    K = rbf_mul(X, X, width_x)
    L = rbf_mul(Y, Y, width_y)
    Kc = ((H @ K) @ H)
    Lc = ((H @ L) @ H)
    testStat = torch.sum(Kc.T * Lc) / n
    return testStat