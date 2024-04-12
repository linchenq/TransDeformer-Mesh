import numpy as np
import torch.nn.functional as nnF
import torch.nn as nn
import torch
import math
# %%
class Sin(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, is_first):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.is_first = is_first
        self.weight = nn.Parameter(torch.zeros(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
        self.reset_parameters()
    
    def reset_parameters(self, bias_min=None, bias_max=None):
        in_dim = self.in_dim
        alpha = self.alpha
        if self.is_first == True:
            self.weight.data.uniform_(-1/in_dim, 1/in_dim)
        else:
            self.weight.data.uniform_(-np.sqrt(6/in_dim)/alpha, np.sqrt(6/in_dim)/alpha)        
        if bias_min is None:
            bias_min = -math.pi / 2
        if bias_max is None:
            bias_max = math.pi / 2
        self.bias.data.uniform_(bias_min, bias_max)

    def forward(self, x, add_bias=True):
        if add_bias == True:
            x1 = nnF.linear(self.alpha*x, self.weight, self.bias)
        else:
            x1 = nnF.linear(self.alpha*x, self.weight)                             
        y = torch.sin(x1)
        return y
# %%
class PositionEncoder(nn.Module):
    def __init__(self, pos_dim, embed_dim, alpha):
        super().__init__()        
        self.pos_dim=pos_dim
        self.embed_dim=embed_dim        
        self.alpha=alpha        
        
        self.weight1 = nn.Parameter(torch.zeros(embed_dim, pos_dim))
        self.bias1 = nn.Parameter(torch.zeros(embed_dim))

        self.weight2 = nn.Parameter(torch.zeros(embed_dim, pos_dim))
        self.bias2 = nn.Parameter(torch.zeros(embed_dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        in_dim = self.pos_dim        
        self.weight1.data.uniform_(-1/in_dim, 1/in_dim)
        self.weight2.data.uniform_(-1/in_dim, 1/in_dim)        
        self.bias1.data.uniform_(0, math.pi)
        self.bias2.data.uniform_(0, math.pi)        

    def forward(self, p, add_bias):
        # p.shape (B, N, D), position, D=pos_dim
        p = self.alpha * p
        if add_bias == True:
            p1 = nnF.linear(p, self.weight1, self.bias1)
            p2 = nnF.linear(p, self.weight2, self.bias2)
        else:
            p1 = nnF.linear(p, self.weight1)
            p2 = nnF.linear(p, self.weight2)

        p_code = [torch.cos(p1), torch.sin(p1), torch.cos(p2), torch.sin(p2)]
        return p_code
# %%
class QueryEncoder(nn.Module):
    def __init__(self, n_heads, embed_dim):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim)
        
        self.mlp1 = nn.Linear(embed_dim, embed_dim)
        self.mlp2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, p_code):
        # x.shape (B, N, E), E = H * C, H = n_heads
        # p_code is from PositionEncoder
        B, N, E = x.shape
        H = self.n_heads
        C = E // H
        x = self.norm(x)
        x1 = self.mlp1(x)   #(B, N, E)
        x2 = self.mlp2(x)   #(B, N, E)        
        out1_cos = x1 * p_code[0]   # cos(w1 * p)
        out1_sin = x1 * p_code[1]   # sin(w1 * p)
        out2_cos = x2 * p_code[2]   # cos(w2 * p)
        out2_sin = x2 * p_code[3]   # sin(w2 * p)
        out = torch.cat([out1_cos.view(B,N,H,C),
                         out1_sin.view(B,N,H,C), 
                         out2_cos.view(B,N,H,C), 
                         out2_sin.view(B,N,H,C)
                         ], dim=-1)
        return out
# %%
class KeyEncoder(nn.Module):
    def __init__(self, n_heads, embed_dim):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp1 = nn.Linear(embed_dim, embed_dim)

    def forward(self, y, p_code):
        # y.shape (B, L, E), E = H * C, H = n_heads
        # p_code is from PositionEncoder
        B, L, E = y.shape
        H = self.n_heads
        C = E // H
        y = self.norm(y)
        y1 = self.mlp1(y)   #(B, L, E)
        out1_cos = y1*p_code[0]
        out1_sin = y1*p_code[1]
        out2_cos = p_code[2]
        out2_sin = p_code[3]
        out = torch.cat([out1_cos.view(B,L,H,C),
                         out1_sin.view(B,L,H,C), 
                         out2_cos.view(B,L,H,C),
                         out2_sin.view(B,L,H,C)
                         ], dim=-1)
        return out 
#%%
class Attention(nn.Module):
    def __init__(self, n_heads, embed_dim, pos_dim, alpha_q, alpha_v, attn_drop=0):
        super().__init__()        
        assert embed_dim % n_heads == 0, 'embed_dim should be divisible by n_heads'
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.pos_dim = pos_dim
        self.alpha_q = alpha_q
        self.alpha_v = alpha_v        
        self.attn_drop = attn_drop
        
        self.P = PositionEncoder(pos_dim, embed_dim, alpha_q)
        self.Q = QueryEncoder(n_heads, embed_dim)
        self.K = KeyEncoder(n_heads, embed_dim)
        self.V = nn.Sequential(nn.LayerNorm(embed_dim),
                               nn.Linear(embed_dim, embed_dim))        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj1 = nn.Linear(embed_dim, embed_dim)
        self.proj2 = nn.Linear(pos_dim*n_heads, embed_dim)

    def forward(self, x, xp, y, yp, attn_mask=None, eps=1e-8):
        # x query/attend to y
        # x.shape (B,N,E), E=embed_dim
        # y.shape (B,L,E)
        # xp: position of x, xp.shape (B,N,D), D=pos_dim
        # yp: position of y, yp.shape (B,L,D)
        # attn_mask: (B, H, N, L) or (B, 1, N, L) or (1, 1, N, L)

        H = self.n_heads
        B, N, E = x.shape
        C = E // H
        L = y.shape[1]
        D = xp.shape[2]
        
        xp_code = self.P(xp, add_bias=True)
        yp_code = self.P(yp, add_bias=False)
        
        q = self.Q(x, xp_code)      #(B, N, H, ?*C)
        q = q.permute(0, 2, 1, 3)   #(B, H, N, ?*C)       
        
        k = self.K(y, yp_code)      #(B, L, H, ?*C)
        k = k.permute(0,2,1,3)      #(B, H, L, ?*C)
        
        scale = (q.shape[-1]) ** -0.5
        # print("scale", scale)
        
        attn = (q @ k.transpose(-2, -1)) * scale
        # attn.shape: (B, H, N, L)
        # print("attn.shape", attn.shape)
        
        attn = attn.softmax(dim=-1)
        
        if attn_mask is not None:
            attn = attn * attn_mask
            eps = min(eps, 1/L)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + eps)
        
        attn = self.attn_drop(attn)
        
        v = self.V(y)   #(B, L, E)
        v = v.reshape(B, L, H, C).permute(0, 2, 1, 3)   #(B, H, L, C)       
      
        out1 = torch.matmul(attn, v)    #(B, H, N, C)
        out1 = out1.permute(0, 2, 1, 3).reshape(B, N, E)
        out1 = self.proj1(out1)     #(B, N, E)
        
        out2 = torch.matmul(attn, yp.view(B, 1, L, D))  #(B, H, N, D)        
        
        out2 = out2 - xp.view(B, 1, N, D)   # relative position
        
        out2 = out2.permute(0, 2, 1, 3).reshape(B, N, H * D)
        out2 = self.proj2(out2)     #(B, N, E) encode relative position
        
        out = out1 + out2   #(B, N, E)
        
        return out, attn
# %%
class ShapeToImageAttentionLayer(nn.Module):
    def __init__(self, n_heads, embed_dim, pos_dim, alpha, attn_drop):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.pos_dim = pos_dim
        self.alpha_q = alpha
        self.alpha_v = alpha        
        self.attn_drop = attn_drop
        
        self.MHA = Attention(n_heads, embed_dim, pos_dim, alpha, alpha, attn_drop)
        self.MLP = nn.Sequential(nn.LayerNorm(embed_dim),
                                 nn.Linear(embed_dim, 2*embed_dim),
                                 nn.GELU(),
                                 nn.Linear(2*embed_dim, embed_dim))

    def forward(self, x, xp, y, yp, attn_mask=None):
        # x  (B, N, E) shape features
        # xp (B, N, 2) shape, real shape, not template
        #              it must be normalized before using it here
        #              so that its range is -1, 1
        # y  (B, L, E) from y_list of ImageSelfAttention
        # yp (B, L, 2) from yp of ImageSelfAttention
        x1, w = self.MHA(x, xp, y, yp, attn_mask=attn_mask)
        x2 = x1 + x
        # add&norm
        x3 = x2 + self.MLP(x2) 
        return x3, w