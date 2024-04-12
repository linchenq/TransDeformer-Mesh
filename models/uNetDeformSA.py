import torch
import torch.nn as nn

from utils.ShapeSelfAttention import ShapeSelfAttentionLayer
from utils.ShapeEmbedding import ObjectShapeEmbedding
# %%
class UNetDeformSA(nn.Module):
    def __init__(self, arg):
        super().__init__()
        self.arg = arg
        n_layers = arg['n_layers']
        n_heads = arg['n_heads']
        embed_dim = arg['embed_dim']
        alpha = arg['alpha']
        attn_drop = arg['attn_drop']
        C_in_low = arg['C_in_low']
        C_in_mid = arg['C_in_mid']
        C_in_high = arg['C_in_high']        
        
        self.shape_embedding0 = ObjectShapeEmbedding(C=C_in_low,
                                                     embed_dim=embed_dim,
                                                     pos_dim=2,
                                                     alpha=alpha)
        
        self.shape_self_attn0 = nn.ModuleList()
        for n in range(0, n_layers):
            self.shape_self_attn0.append(ShapeSelfAttentionLayer(n_heads=n_heads, 
                                                                 embed_dim=embed_dim, 
                                                                 pos_dim=2,
                                                                 alpha=alpha,
                                                                 attn_drop=attn_drop))
        
        self.attn_out0 = nn.Sequential(nn.LayerNorm(embed_dim),
                                       nn.Linear(embed_dim, embed_dim),
                                       nn.GELU(),
                                       nn.Linear(embed_dim, 2)
                                       )
        
        self.shape_embedding1 = ObjectShapeEmbedding(C=C_in_mid,
                                                     embed_dim=embed_dim,
                                                     pos_dim=2,
                                                     alpha=alpha)
        
        self.proj1 = nn.Sequential(nn.LayerNorm(embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
        
        self.shape_self_attn1 = nn.ModuleList()
        for n in range(0, n_layers):
            self.shape_self_attn1.append(ShapeSelfAttentionLayer(n_heads=n_heads, 
                                                                 embed_dim=embed_dim, 
                                                                 pos_dim=2,
                                                                 alpha=alpha,
                                                                 attn_drop=attn_drop))
        
        self.attn_out1 = nn.Sequential(nn.LayerNorm(embed_dim),
                                       nn.Linear(embed_dim, embed_dim),
                                       nn.GELU(),
                                       nn.Linear(embed_dim, 2)
                                       )

        self.shape_embedding2 = ObjectShapeEmbedding(C=C_in_high,
                                                     embed_dim=embed_dim,
                                                     pos_dim=2,
                                                     alpha=alpha)
        
        self.proj2 = nn.Sequential(nn.LayerNorm(embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
        
        self.shape_self_attn2 = nn.ModuleList()
        for n in range(0, n_layers):
            self.shape_self_attn2.append(ShapeSelfAttentionLayer(n_heads=n_heads, 
                                                                 embed_dim=embed_dim, 
                                                                 pos_dim=2,
                                                                 alpha=alpha,
                                                                 attn_drop=attn_drop))
        
        self.attn_out2 = nn.Sequential(nn.LayerNorm(embed_dim),
                                       nn.Linear(embed_dim, embed_dim),
                                       nn.GELU(),
                                       nn.Linear(embed_dim, 2)
                                       )        
        
    def forward(self, obj_init, template, feature_high, feature_mid, feature_low, 
                u0_flag=1, u1_flag=1, u2_flag=1):
        # obj_init: (B, N, 2) in the 512 x 512 image space
        # template: (B, N, 2) in the 512 x 512 image space
        # feature: (B, C, H, W) feature map from U-net
        
        # Please note that: 511 = 512 - 1 is the original mesh/image space for coordinates of the template/obj
        template = -1 + template / (511/2) # normalization to range: -1 to 1
               
        x0 = self.shape_embedding0(obj_init, template, feature_low, self.arg['beta'])  # (B, N, E)
        for n in range(0, self.arg['n_layers']): 
            x0, w0 = self.shape_self_attn0[n](x0, template)
        u0 = self.attn_out0(x0)     # (B, N, 2)
        if u0_flag == 1:
            obj0 = obj_init + u0
        else:
            obj0 = obj_init
        
        x1 = self.shape_embedding1(obj0, template, feature_mid, self.arg['beta'])   # (B, N, E)
        x1 = x1 + self.proj1(x0)
        for n in range(0, self.arg['n_layers']): 
            x1, w1 = self.shape_self_attn1[n](x1, template)
        u1 = self.attn_out1(x1)     # (B, N, 2)
        if u1_flag == 1:
            obj1 = obj0 + u1
        else:
            obj1 = obj0
        
        x2 = self.shape_embedding2(obj1, template, feature_high, self.arg['beta'])  # (B, N, E)
        x2 = x2 + self.proj2(x1)
        for n in range(0, self.arg['n_layers']): 
            x2, w2 = self.shape_self_attn2[n](x2, template)
        u2 = self.attn_out2(x2)     # (B, N, 2)
        if u2_flag == 1:
            obj2 = obj1 + u2
        else:
            obj2 = obj1
               
        return obj0, obj1, obj2                         