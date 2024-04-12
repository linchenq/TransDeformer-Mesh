import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
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
            bias_min = -math.pi/2
        if bias_max is None:
            bias_max = math.pi/2
        self.bias.data.uniform_(bias_min, bias_max)

    def forward(self, x, add_bias=True):
        if add_bias == True:
            x1 = nnF.linear(self.alpha*x, self.weight, self.bias)
        else:
            x1 = nnF.linear(self.alpha*x, self.weight)                             
        y = torch.sin(x1)
        return y
# %%
class PositionEmbedding(nn.Module):
    def __init__(self, H, W, embed_dim, alpha, use_sin_embed):
        super().__init__()
        self.H = H
        self.W = W
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.use_sin_embed = use_sin_embed
        
        self.projector = nn.Linear(embed_dim, embed_dim)
        if use_sin_embed == True:
            self.pos_encoder = nn.Sequential(Sin(2, embed_dim, alpha, is_first=True),
                                             Sin(embed_dim, embed_dim, 1, is_first=False), 
                                             nn.Linear(embed_dim, embed_dim))
            
            yy = torch.arange(0, H)
            xx = torch.arange(0, W) 
            grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
            grid_y = grid_y.reshape(1,H*W,1).to(torch.float32)
            grid_x = grid_x.reshape(1,H*W,1).to(torch.float32)
            grid = torch.cat([grid_x/(W-1), grid_y/(H-1)], dim=-1) # grid.shape (1, H*W, 2)
            grid = 2 * grid - 1
            self.register_buffer("grid", grid)
        else:
            self.pos_embed = nn.Parameter(0.01*torch.randn((1,H*W,embed_dim)))

    def forward(self, x):
        # x.shape (B, H*W, C), C=embed_dim
        # assume linear projection has been applied to x
        x = self.projector(x)
        if self.use_sin_embed == True:            
            pos = self.pos_encoder(self.grid) #pos.shape (1,H*W,C)
        else:
            pos = self.pos_embed
        y = x + pos # y.shape (B,H*W,C)
        return y 
#%%
class Attention(nn.Module):
    def __init__(self, n_heads, embed_dim, attn_drop=0):
        super().__init__()        
        assert embed_dim % n_heads == 0, 'embed_dim should be divisible by n_heads'
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.attn_drop = attn_drop
        
        self.Q = nn.Sequential(nn.LayerNorm(embed_dim),
                               nn.Linear(embed_dim, embed_dim))   
        self.K = nn.Sequential(nn.LayerNorm(embed_dim),
                               nn.Linear(embed_dim, embed_dim))   
        self.V = nn.Sequential(nn.LayerNorm(embed_dim),
                               nn.Linear(embed_dim, embed_dim))        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, y, attn_mask=None, eps=1e-8):
        # x query/attend to y
        # x.shape (B, N, E), E = embed_dim
        # y.shape (B, L, E)
        # xp: position of x, xp.shape (B, N, D), D = pos_dim
        # yp: position of y, yp.shape (B, L, D)
        # attn_mask: (B, H, N, L) or (B, 1, N, L) or (1, 1, N, L)
        # attn_mask[:.:.n,l] = 0 or 1
        #--------------------------------------------
        H = self.n_heads
        B, N, E = x.shape
        C = E // H
        L = y.shape[1]
        
        q = self.Q(x)   #(B, N, E)
        q = q.reshape(B, N, H, C).permute(0, 2, 1, 3)   #(B, H, N, C)
        
        k = self.K(y)   #(B, L, E)
        k = k.reshape(B, L, H, C).permute(0, 2, 1, 3)   #(B,H,L,C)
        
        scale = (q.shape[-1]) ** -0.5
        # print("scale", scale)
        
        attn = (q @ k.transpose(-2, -1))*scale
        # attn.shape (B, H, N, L)
        # print("attn.shape", attn.shape)
        
        attn = attn.softmax(dim=-1)
        
        if attn_mask is not None:
            attn = attn * attn_mask
            eps = min(eps, 1/L)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + eps)
        
        attn = self.attn_drop(attn)        
        
        v = self.V(y)     #(B,L,E)
        v = v.reshape(B, L, H, C).permute(0, 2, 1, 3)   #(B, H, L, C)
        
        out = torch.matmul(attn, v)     #(B, H, N, C)
        out = out.permute(0, 2, 1, 3).reshape(B, N, E)
        out = self.proj(out)    #(B, N, E)        
        
        return out, attn
#%%
class AttentionLayer(nn.Module):
    def __init__(self, n_heads, embed_dim, pos_dim, attn_drop):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.pos_dim=pos_dim
        self.attn_drop=attn_drop
        
        self.MHA = Attention(n_heads, embed_dim, attn_drop)
        self.MLP = nn.Sequential(nn.LayerNorm(embed_dim),
                                 nn.Linear(embed_dim, 2*embed_dim),
                                 nn.GELU(),
                                 nn.Linear(2*embed_dim, embed_dim))

    def forward(self, x, y, attn_mask=None):
        x1, w = self.MHA(x, y, attn_mask=attn_mask)
        x2 = x1 + x
        # add & norm
        x3 = x2 + self.MLP(x2) 
        return x3, w
# %%
# nn.Sequential is supported: This layer can be stacked
class SelfAttentionLayer(nn.Module):
    def __init__(self, n_heads, embed_dim, attn_drop):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.attn_drop = attn_drop
        
        self.MHA = Attention(n_heads, embed_dim, attn_drop)
        self.MLP = nn.Sequential(nn.LayerNorm(embed_dim),
                                 nn.Linear(embed_dim, 2*embed_dim),
                                 nn.GELU(),
                                 nn.Linear(2*embed_dim, embed_dim))

    def forward(self, x_w_m):
        x, w, m = x_w_m
        x1, w1 = self.MHA(x, x, m)
        x2 = x1 + x
        # add & norm
        x3 = x2 + self.MLP(x2)
        w = w + w1
        return x3, w, m
# %%
# Patch embedding and Position embedding are included
class AttentionBlock(nn.Module):
    def __init__(self, H_in, W_in, C_in, H_out, W_out, C_out,
                 patch_size, embed_dim, n_heads, attn_drop, n_layers, alpha, use_sin_embed=False):
        super().__init__()
        self.H_in = H_in
        self.W_in = W_in
        self.C_in = C_in
        self.H_out = H_out
        self.W_out = W_out
        self.C_out = C_out
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.attn_drop = attn_drop
        self.n_layers = n_layers
        self.alpha = alpha
        self.use_sin_embed = use_sin_embed
        
        self.patch_layer = nn.Conv2d(C_in, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.patch_norm = nn.LayerNorm(embed_dim)        
        
        if isinstance(patch_size, int):
            H1 = H_in // patch_size
            W1 = W_in // patch_size
        else:
            H1 = H_in // patch_size[0]
            W1 = W_in // patch_size[1]
        self.H1 = H1
        self.W1 = W1
        self.pos_emb = PositionEmbedding(H1, W1, embed_dim, alpha, use_sin_embed=use_sin_embed)
        
        attn_layers = []
        for _ in range(n_layers):
            attn_layers.append(SelfAttentionLayer(n_heads, embed_dim, attn_drop))
        self.attn_layers = nn.Sequential(*attn_layers)
        self.attn_out = nn.Sequential(nn.LayerNorm(embed_dim),
                                      nn.Linear(embed_dim, C_out))

    def forward(self, x):
        # x.shape (B, C, H, W)
        x1 = self.patch_layer(x)        #(B, E, H1, W1) 
        x1 = x1.permute(0, 2, 3, 1)     #(B, H1, W1, E) 
        x1 = x1.reshape(-1, self.H1 * self.W1, self.embed_dim)  #(B, L, E), L = H1 * W1        
        x1 = self.patch_norm(x1)
        x1 = self.pos_emb(x1)
        
        w = 0       # attn_weight
        m = None    # mask
        x2, w, m = self.attn_layers((x1, w, m))
        w = w / self.n_layers
        y = self.attn_out(x2)   #(B, L, C_out)
        y = y.permute(0, 2, 1)  #(B, C_out, L), L = H1 * W1
        y = y.reshape(-1, self.C_out, self.H1, self.W1) 
        
        if (self.H1 != self.H_out) or (self.W1 != self.W_out):
            y = nnF.interpolate(y, size=(self.H_out, self.W_out), mode='bilinear', align_corners=False)       
        # y.shape (B, C_out, H_out, W_out)
        return y, w
# %%
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        hid_dim = 2 * max([in_channels, out_channels])
        self.conv = nn.Sequential(nn.Conv2d(in_channels, hid_dim, kernel_size, stride, padding),
                                  nn.GroupNorm(1, hid_dim, affine=False),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Conv2d(hid_dim, hid_dim, kernel_size=1, stride=1, padding=0),
                                  nn.GroupNorm(1, hid_dim, affine=False),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Conv2d(hid_dim, out_channels, kernel_size=3, stride=1, padding=1))
        self.res_path = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        # x.shape (B, C, H, W), C = in_channels        
        y = self.res_path(x) + self.conv(x)     #(B, out_channels, H_new, W_new)
        return y
# %%
# Spatial Contract Norm functions and class
def spatial_contrast_norm(x, kernel_size=None, eps=1e-5):
    # x.shape (B, C, H, W)
    B, C, H, W = x.shape
    if kernel_size is None:
        kernel_size = (H // 8, W // 8)
        if H <= 8 and W <= 8:
            x = nnF.group_norm(x, 1)
            return x
    if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
        pad_h = kernel_size[0] // 2
        pad_w = kernel_size[1] // 2
        h = 2 * pad_h + 1
        w = 2 * pad_w + 1
    else:
        pad_h = pad_w = kernel_size // 2
        h = w = 2 * pad_h + 1
    padding = (pad_w, pad_w, pad_h, pad_h)
    x_mean = x.mean(dim=1, keepdim=True)    
    if h > 1 or w > 1:        
        x_mean = nnF.pad(x_mean, padding, 'reflect')
        x_mean = nnF.avg_pool2d(x_mean, kernel_size=(h, w), padding=0, stride=1)    
    x_var = (x - x_mean)**2
    x_var = x_var.mean(dim=1, keepdim=True)
    if h > 1 or w > 1:
        x_var = nnF.pad(x_var, padding, 'reflect')
        x_var = nnF.avg_pool2d(x_var, kernel_size=(h, w), padding=0, stride=1)
    x_std = (x_var + eps).sqrt()    
    y = (x - x_mean) / x_std
    return y

class SpatialContrastNorm(nn.Module):
    def __init__(self, kernel_size=None, eps=1e-5):        
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps
        
    def forward(self, x):
        y = spatial_contrast_norm(x, self.kernel_size, self.eps)
        return y
# %%
class Block(nn.Module):
    def __init__(self, H_in, W_in, C_in, H_out, W_out, C_out,
                 kernel_size, stride, padding, #conv parameters
                 patch_size, embed_dim, n_heads, dropout, n_layers, alpha #attn parameters
                 ):
        super().__init__()
        self.n_layers = n_layers        
        self.norm0 = nn.GroupNorm(1, C_in, affine=False)
        self.cnn_block = CNNBlock(C_in, C_out, kernel_size, stride, padding)
        if n_layers > 0:
            self.attn_block = AttentionBlock(H_in, W_in, C_in, H_out, W_out, C_out,
                                             patch_size, embed_dim, n_heads, dropout, n_layers, alpha)        
        self.out = nn.Sequential(nn.GroupNorm(1, C_out, affine=False),
                                 nn.LeakyReLU(inplace=True))
                
    def forward(self, x, a1=1, a2=1):
        x = self.norm0(x)
        y1 = 0
        if a1 != 0:
            y1 = self.cnn_block(x)
        y2 = 0
        w = 0
        if a2 != 0 and self.n_layers > 0:
            y2, w = self.attn_block(x)
        y = a1 * y1 + a2 * y2
        y = self.out(y)
        return y, w
# %%
class MergeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, scale_factor):
        super().__init__()        
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.norm1 = nn.GroupNorm(1, in_channels, affine=False)
        self.norm2 = nn.GroupNorm(1, skip_channels, affine=False)
        self.proj1 = nn.Conv2d(in_channels, out_channels,   kernel_size=1, stride=1, padding=0)
        self.proj2 = nn.Conv2d(skip_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.out = nn.Sequential(nn.GroupNorm(1, out_channels, affine=False),
                                 nn.LeakyReLU(inplace=True))

    def forward(self, skip, x, a1=1, a2=1):
        # x.shape (B, C1, H/4, W/4), C1 = in_channels
        # skip.shape (B, C2, H, W), C2 = skip_channels
        x = self.up(x) #(B, C1, H, W)
        x = self.norm1(x)
        skip = self.norm2(skip)
        x = self.proj1(x)
        skip = self.proj2(skip)
        y = a1 * x + a2 * skip
        y = self.out(y)
        # y.shape (B, C, H, W), C = out_channels
        return y
# %%
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, upsampling):
        super().__init__()
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.conv = nn.Sequential(nn.GroupNorm(1, in_channels, affine=False),
                                  nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                            kernel_size=5, stride=1, padding=2),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=1, stride=1, padding=0))
    def forward(self, x):
        y = self.conv(self.upsampling(x))
        return y             
# %%
class UNetBackbone(nn.Module):
    def __init__(self, num_classes, 
                       patch_size,
                       embed_dim,
                       n_heads,
                       dropout,
                       n_layers,
                       alpha,
                       use_cnn,
                       use_attn):        
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.n_layers = n_layers
        self.alpha = alpha
        self.patch_size = patch_size
        self.use_cnn = use_cnn
        self.use_attn = use_attn        
        
        ps = self.patch_size
        nl = self.n_layers
        if isinstance(nl, int):
            nl = [nl, nl, nl, nl]
        
        self.norm0 = SpatialContrastNorm(kernel_size=(512//8, 512//8))

        self.T0a = Block(H_in=512,  W_in=512,  C_in=1,
                         H_out=512, W_out=512, C_out=32,
                         kernel_size=5, stride=1, padding=2,
                         patch_size=ps[0], embed_dim=self.embed_dim, 
                         n_heads=self.n_heads, dropout=self.dropout, n_layers=nl[0], alpha=self.alpha)
                
        self.T1a = Block(H_in=512,  W_in=512,  C_in=32, 
                         H_out=128, W_out=128, C_out=128,
                         kernel_size=5, stride=4, padding=2,
                         patch_size=ps[1], embed_dim=self.embed_dim,
                         n_heads=self.n_heads, dropout=self.dropout, n_layers=nl[1], alpha=self.alpha)
       
        self.T2a = Block(H_in=128, W_in=128, C_in=128, 
                         H_out=32, W_out=32, C_out=512,
                         kernel_size=5, stride=4, padding=2,
                         patch_size=ps[2], embed_dim=self.embed_dim,
                         n_heads=self.n_heads, dropout=self.dropout, n_layers=nl[2], alpha=self.alpha)
        
        self.T3a = Block(H_in=32, W_in=32, C_in=512, 
                         H_out=16, W_out=16, C_out=512,
                         kernel_size=3, stride=2, padding=1,
                         patch_size=ps[3], embed_dim=self.embed_dim,
                         n_heads=self.n_heads, dropout=self.dropout, n_layers=nl[3], alpha=self.alpha)
        
        self.T3b = nn.Identity()

        self.M2 = MergeLayer(in_channels=512, out_channels=512,
                             skip_channels=512, scale_factor=2)
        
        self.T2b = Block(H_in=32,   W_in=32,  C_in=512, 
                         H_out=32,  W_out=32, C_out=128,
                         kernel_size=3, stride=1, padding=1,
                         patch_size=ps[-1], embed_dim=self.embed_dim,
                         n_heads=self.n_heads, dropout=self.dropout, n_layers=nl[-1], alpha=self.alpha)
        
        self.M1 = MergeLayer(in_channels=128, out_channels=128,
                             skip_channels=128, scale_factor=4)
        
        self.T1b = Block(H_in=128,  W_in=128,  C_in=128,
                         H_out=128, W_out=128, C_out=32,
                         kernel_size=5, stride=1, padding=2,
                         patch_size=ps[-2], embed_dim=self.embed_dim,
                         n_heads=self.n_heads, dropout=self.dropout, n_layers=nl[-2], alpha=self.alpha)
        
        self.M0 = MergeLayer(in_channels=32, out_channels=32,
                             skip_channels=32, scale_factor=4)
        
        self.T0b = Block(H_in=512,  W_in=512,  C_in=32, 
                         H_out=512, W_out=512, C_out=32,
                         kernel_size=5, stride=1, padding=2,
                         patch_size=ps[-3], embed_dim=self.embed_dim,
                         n_heads=self.n_heads, dropout=self.dropout, n_layers=nl[-3], alpha=self.alpha)
        
        self.seghead = SegmentationHead(32, self.num_classes, upsampling=1)

    def forward(self, x, return_attn_weight=False):
        
        a1 = 1
        a2 = 1
        if self.use_cnn != 1:
            a1 = 0
        if self.use_attn != 1:
            a2 = 0
               
        x = self.norm0(x)
        
        # print('x', x.shape)
        t0a, w0a = self.T0a(x, a1=a1, a2=a2)
        #p rint("t0a.shape", t0a.shape)

        t1a, w1a = self.T1a(t0a, a1=a1, a2=a2)
        # print("t1a.shape", t1a.shape)

        t2a, w2a = self.T2a(t1a, a1=a1, a2=a2)
        # print("t2a.shape", t2a.shape)

        t3a, w3a = self.T3a(t2a, a1=a1, a2=a2)
        # print("t3a.shape", t3a.shape)
        # t3a.shape: (B, C, H, W)

        t3b = self.T3b(t3a)
        # print("t3b.shape", t3b.shape)

        m2 = self.M2(t2a, t3b)
        # print("m2.shape", m2.shape)

        t2b, w2b = self.T2b(m2, a1=a1, a2=a2)
        # print("t2b.shape", t2b.shape)

        m1 = self.M1(t1a, t2b)
        # print("m1.shape", m1.shape)

        t1b, w1b = self.T1b(m1, a1=a1, a2=a2)
        # print("t1b.shape", t1b.shape)

        m0 = self.M0(t0a, t1b)
        # print("m0.shape", m0.shape)

        t0b, w0b = self.T0b(m0, a1=a1, a2=a2)
        # print("t0b.shape", t0b.shape)

        out = self.seghead(t0b)
        # print("out",out.shape)

        return out, [t0a, t1a, t2a, t3a, t3b, t2b, t1b, t0b]
# %%
if __name__ == '__main__':
    model = UNetBackbone(num_classes=12, 
                         patch_size=[16, 16, 4, 1],
                         embed_dim=512,
                         n_heads=16,
                         dropout=0,
                         n_layers=[0, 0, 0, 0],
                         alpha=100, 
                         use_cnn=1, 
                         use_attn=1)
    x = torch.rand((1, 1, 512, 512))
    out, ys = model(x)
    for y in ys:
        print(y.shape)
        