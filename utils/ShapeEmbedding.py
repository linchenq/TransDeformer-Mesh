import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as nnF
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
def SampleImage(image, point):
    # image: (B, C, H, W),  H is y-axis, W is x-axis
    # point: (B, N, 2), point[:, :, 0] is x, point[:, :, 1] is y
    # point is defined in the 512 x 512 image space
    
    point = -1 + point / (511 / 2) #range: -1 to 1
    # print('1', point.shape)
    # point.shape:  B x N x 2 => B x N x 1 x 2 to use grid_sample
    point = point.view(point.shape[0], point.shape[1], 1, 2)
    #print('2', point.shape)
    output = nn.functional.grid_sample(input=image, grid=point,
                                       mode='bilinear', padding_mode='zeros', align_corners=True)
    # print('3', output.shape)
    # output.shape: B x C x N x 1 => B x C x N
    output = output.view(output.shape[0], output.shape[1], output.shape[2])
    # print('4', output.shape)   
    # output.shape: B x C x N => B x N x C
    output = output.permute(0, 2, 1)
    # print('5', output.shape)
    return output
# %%
class TemplateShapeEmbedding(nn.Module):
    def __init__(self, embed_dim, pos_dim, alpha):
        super().__init__()
        self.pos = nn.Sequential(Sin(pos_dim, embed_dim, alpha, True),
                                 Sin(embed_dim, embed_dim, 1, False),
                                 Sin(embed_dim, embed_dim, 1, False),
                                 nn.Linear(embed_dim, embed_dim))        

    def forward(self, template):
        # template: (B, N, D), template shape, D = pos_dim=2, normalized in some models
        pos_emb = self.pos(template)    # (B, N, E)    
        return pos_emb
# %%
class ObjectShapeEmbedding(nn.Module):
    def __init__(self, C, embed_dim, pos_dim, alpha):
        super().__init__()
        self.pos = TemplateShapeEmbedding(embed_dim, pos_dim, alpha)   
        self.proj = nn.Sequential(nn.Linear(C, embed_dim),
                                  nn.LayerNorm(embed_dim),
                                  nn.Linear(embed_dim, embed_dim)) 

    def forward(self, obj, template, feature, beta):
        # obj: (B, N, 2) in the 512x512 image space
        # template: (B, N, 2), normalized in range of -1 to 1 in ShapeDeformNet 
        # feature: (B,C,H,W) feature map from U-net
        # beta is 0 (feature is useless) or 1 (feature is useful)
        pos_emb = self.pos(template)    #(B, N, E)
        if beta == 0:
            return pos_emb        
        
        img_emb = SampleImage(image=feature, point=obj)     #(B, N, C)    
        img_emb = self.proj(img_emb)    #(B, N, E)        
        obj_emb = pos_emb + img_emb
        return obj_emb      #(B, N, E)
