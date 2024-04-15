import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import torch.nn.functional as nnF
import torch.nn as nn
import torch

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
        y = self.res_path(x) + self.conv(x)     # (B, out_channels, H_new, W_new)
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
                 ):
        super().__init__()       
        self.norm0 = nn.GroupNorm(1, C_in, affine=False)
        self.cnn_block = CNNBlock(C_in, C_out, kernel_size, stride, padding)
        self.out = nn.Sequential(nn.GroupNorm(1, C_out, affine=False),
                                 nn.LeakyReLU(inplace=True))
                
    def forward(self, x):
        x = self.norm0(x)
        y = self.cnn_block(x)
        y = self.out(y)
        return y
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

    def forward(self, skip, x):
        # x.shape (B, C1, H/4, W/4), C1 = in_channels
        # skip.shape (B, C2, H, W), C2 = skip_channels
        x = self.up(x) #(B, C1, H, W)
        x = self.norm1(x)
        x = self.proj1(x)
        y = x
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
    def __init__(self, num_classes):        
        super().__init__()
        self.num_classes = num_classes
     
        self.norm0 = SpatialContrastNorm(kernel_size=(512//8, 512//8))

        self.T0a = Block(H_in=512,  W_in=512,  C_in=1,
                         H_out=512, W_out=512, C_out=32,
                         kernel_size=5, stride=1, padding=2)
                
        self.T1a = Block(H_in=512,  W_in=512,  C_in=32, 
                         H_out=128, W_out=128, C_out=128,
                         kernel_size=5, stride=4, padding=2)
       
        self.T2a = Block(H_in=128, W_in=128, C_in=128, 
                         H_out=32, W_out=32, C_out=512,
                         kernel_size=5, stride=4, padding=2)
        
        self.T3a = Block(H_in=32, W_in=32, C_in=512, 
                         H_out=16, W_out=16, C_out=512,
                         kernel_size=3, stride=2, padding=1)
        
        self.T3b = nn.Identity()

        self.M2 = MergeLayer(in_channels=512, out_channels=512,
                             skip_channels=512, scale_factor=2)
        
        self.T2b = Block(H_in=32,   W_in=32,  C_in=512, 
                         H_out=32,  W_out=32, C_out=128,
                         kernel_size=3, stride=1, padding=1)
        
        self.M1 = MergeLayer(in_channels=128, out_channels=128,
                             skip_channels=128, scale_factor=4)
        
        self.T1b = Block(H_in=128,  W_in=128,  C_in=128,
                         H_out=128, W_out=128, C_out=32,
                         kernel_size=5, stride=1, padding=2)
        
        self.M0 = MergeLayer(in_channels=32, out_channels=32,
                             skip_channels=32, scale_factor=4)
        
        self.T0b = Block(H_in=512,  W_in=512,  C_in=32, 
                         H_out=512, W_out=512, C_out=32,
                         kernel_size=5, stride=1, padding=2)
        
        self.seghead = SegmentationHead(32, self.num_classes, upsampling=1)

    def forward(self, x, return_attn_weight=False):
               
        x = self.norm0(x)
        
        # print('x', x.shape)
        t0a = self.T0a(x)
        #p rint("t0a.shape", t0a.shape)

        t1a = self.T1a(t0a)
        # print("t1a.shape", t1a.shape)

        t2a  = self.T2a(t1a)
        # print("t2a.shape", t2a.shape)

        t3a = self.T3a(t2a)
        # print("t3a.shape", t3a.shape)
        # t3a.shape: (B, C, H, W)

        t3b = self.T3b(t3a)
        # print("t3b.shape", t3b.shape)

        m2 = self.M2(t2a, t3b)
        # print("m2.shape", m2.shape)

        t2b = self.T2b(m2)
        # print("t2b.shape", t2b.shape)

        m1 = self.M1(t1a, t2b)
        # print("m1.shape", m1.shape)

        t1b = self.T1b(m1)
        # print("t1b.shape", t1b.shape)

        m0 = self.M0(t0a, t1b)
        # print("m0.shape", m0.shape)

        t0b = self.T0b(m0)
        # print("t0b.shape", t0b.shape)

        out = self.seghead(t0b)
        # print("out",out.shape)

        return out, [t0a, t1a, t2a, t3a, t3b, t2b, t1b, t0b]
# %%
if __name__ == '__main__':
    model = UNetBackbone(num_classes=12)
    # model.load_state_dict(torch.load("../checkpoints/backbone_checkpoint.pt", map_location='cpu')['model_state_dict'])
    x = torch.rand((1, 1, 512, 512))
    out, ys = model(x)
    for y in ys:
        print(y.shape)
        