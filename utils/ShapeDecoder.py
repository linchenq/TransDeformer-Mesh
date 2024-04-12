import torch
import torch.nn as nn
import numpy as np
# %%
def init_weight(linear, alpha, is_first):
    in_dim = linear.in_features
    if is_first == True:
        linear.weight.data.uniform_(-1/in_dim, 1/in_dim)
    else:
        linear.weight.data.uniform_(-np.sqrt(6/in_dim)/alpha, np.sqrt(6/in_dim)/alpha)
# %%
class Sin(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, is_first):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.is_first = is_first
        self.linear1 = nn.Linear(in_dim, out_dim)
        init_weight(self.linear1, alpha, is_first)
    def forward(self, x):
        x1 = self.linear1(self.alpha * x)
        y = torch.sin(x1)
        return y
# %%
class BaseNet(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, x_scale, alpha, gamma, y_dim):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.x_scale = x_scale
        self.alpha = alpha
        self.gamma = gamma
        self.y_dim = y_dim
        self.layer0 = Sin(x_dim, h_dim, x_scale, True)
        
        layer1 = []
        for n in range(0, n_layers):
            layer1.append(Sin(h_dim, h_dim, alpha, False))
        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Linear(h_dim, y_dim)
        self.layer2.weight.data *= gamma
        self.layer2.bias.data *= 0

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        y = self.layer2(x1)
        return y
#%%
class Decoder(nn.Module):
    def __init__(self, embed_dim, h_dim, n_layers, alpha, gamma):
        super().__init__()
        self.BaseNet = BaseNet
        self.embed_dim = embed_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.alpha = alpha
        self.netA = BaseNet(x_dim=2, h_dim=h_dim, n_layers=n_layers, x_scale=1, alpha=alpha, gamma=gamma, y_dim=2*h_dim)
        self.netB = nn.Sequential(nn.LayerNorm(embed_dim),
                                  nn.Linear(embed_dim, h_dim),
                                  nn.Softplus(),
                                  nn.Linear(h_dim, h_dim),
                                  nn.Softplus())
    def forward(self, x, c):
        # x.shape (B, N, 2) template (normalzied between -1 and 1)
        # c.shape (B, N, E) code, embed_dim=E
        B = x.shape[0]
        N = x.shape[1]
        y1 = self.netA(x)
        y2 = self.netB(c)
        y1 = y1.view(B * N, 2, self.h_dim)
        y2 = y2.view(B * N, self.h_dim, 1)
        y = torch.bmm(y1, y2)
        y = y.view(B, N, 2)
        return y
#%%
if __name__ == '__main__':
    x=torch.rand(10, 1080, 2)
    c=torch.rand(10, 1080, 512)
    decoder=Decoder(512, 512, 4, 100, 0.01)
    y=decoder(x, c)
    print(y.shape)
    print(y.abs().mean().item(), y.abs().max().item())


