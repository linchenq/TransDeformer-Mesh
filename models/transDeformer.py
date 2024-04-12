import torch
import torch.nn as nn

from utils.ShapeSelfAttention import ShapeSelfAttentionLayer
from utils.ImageSelfAttention import ImageSelfAttention
from utils.ShapeToImageAttention import ShapeToImageAttentionLayer
from utils.ShapeEmbedding import ObjectShapeEmbedding
from utils.SpatialContrastNorm import SpatialContrastNorm
# %%
class CNNBlock(nn.Module):
    def __init__(self, C_in, C_out, H_in, W_in, kernel_size, stride, padding):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        hid_dim = 2 * max([C_in, C_out])
        self.norm = SpatialContrastNorm(kernel_size=(H_in//8, W_in//8))
        self.conv = nn.Sequential(nn.Conv2d(C_in, hid_dim, kernel_size, stride, padding),
                                  nn.GroupNorm(1, hid_dim, affine=False),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Conv2d(hid_dim, hid_dim, kernel_size=1, stride=1, padding=0),
                                  nn.GroupNorm(1, hid_dim, affine=False),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Conv2d(hid_dim, C_out, kernel_size=3, stride=1, padding=1))
        self.res_path = nn.Conv2d(C_in, C_out, kernel_size, stride, padding)
        
    def forward(self, x):
        # x.shape (B, C, H, W), C = in_channels        
        x = self.norm(x)
        y = self.res_path(x) + self.conv(x)     #(B, out_channels, H_new, W_new)
        return y
#%%
class ImageFeatureLayer(nn.Module):    
    def __init__(self, ):
        super().__init__()
        self.layer0 = CNNBlock(C_in=1,  C_out=32, H_in=512, W_in=512, kernel_size=5, stride=1, padding=2)
        self.layer1 = CNNBlock(C_in=32, C_out=128, H_in=128, W_in=128, kernel_size=5, stride=4, padding=2)
    def forward(self, x):
        # x.shape (B, 1, 512, 512)
        x0 = self.layer0(x)  # x0.shape (B, 32, 512, 512)
        x1 = self.layer1(x0) # x1.shape (B, 128, 128, 128)
        return x0, x1
#%%
class TransDeformer(nn.Module):
    def __init__(self, arg):
        super().__init__()
        self.arg = arg
        n_layers = arg['n_layers']
        n_heads = arg['n_heads']
        embed_dim = arg['embed_dim']
        alpha = arg['alpha']
        attn_drop = arg['attn_drop']
        patch_size = arg['patch_size']
        
        self.image_feature = ImageFeatureLayer()
        
        self.shape_embedding = ObjectShapeEmbedding(C=128,
                                                    embed_dim=embed_dim,
                                                    pos_dim=2,
                                                    alpha=alpha)
        
        self.shape_self_attn = nn.ModuleList()
        for n in range(0, n_layers + 2):
            self.shape_self_attn.append(ShapeSelfAttentionLayer(n_heads=n_heads, 
                                                                embed_dim=embed_dim, 
                                                                pos_dim=2,
                                                                alpha=alpha,
                                                                attn_drop=attn_drop))
        self.image_self_attn = ImageSelfAttention(H_in=128, 
                                                  W_in=128, 
                                                  C_in=128, 
                                                  patch_size=patch_size,
                                                  embed_dim=embed_dim,
                                                  n_heads=n_heads,
                                                  attn_drop=attn_drop,
                                                  n_layers=n_layers, 
                                                  alpha=alpha)
        
        self.shape_to_image_attn = nn.ModuleList()
        for n in range(0, n_layers):
            self.shape_to_image_attn.append(ShapeToImageAttentionLayer(n_heads=n_heads, 
                                                                       embed_dim=embed_dim, 
                                                                       pos_dim=2,
                                                                       alpha=alpha,
                                                                       attn_drop=attn_drop))                
                       
        self.out = nn.Sequential(nn.LayerNorm(embed_dim),
                                 nn.Linear(embed_dim, embed_dim),
                                 nn.Softplus(),
                                 nn.Linear(embed_dim, embed_dim),
                                 nn.Softplus(),
                                 nn.Linear(embed_dim, 2))
        
        self.shape_embedding1 = ObjectShapeEmbedding(C=32,
                                                    embed_dim=embed_dim,
                                                    pos_dim=2,
                                                    alpha=alpha)
        
        self.shape_self_attn1 = nn.ModuleList()
        for n in range(0, n_layers):
            self.shape_self_attn1.append(ShapeSelfAttentionLayer(n_heads=n_heads, 
                                                                 embed_dim=embed_dim, 
                                                                 pos_dim=2,
                                                                 alpha=alpha,
                                                                 attn_drop=attn_drop))
        
        self.out1 = nn.Sequential(nn.LayerNorm(embed_dim),
                                  nn.Linear(embed_dim, embed_dim),
                                  nn.Softplus(),
                                  nn.Linear(embed_dim, embed_dim),
                                  nn.Softplus(),
                                  nn.Linear(embed_dim, 2),
                                  nn.Tanh())
            
    def forward(self, obj_init, template, img):
        # obj_init: (B, N, 2) in the 512x512 image space
        # template: (B, N, 2) in the 512x512 image space
        # img: (B,1,512,512) image
        
        feature512, feature128 = self.image_feature(img)
        template = -1 + template / (511/2)  # normalization to range: -1 to 1
             
        x0 = self.shape_embedding(obj_init, template, feature128, 1)    #(B, N, E)       
        xp = -1 + obj_init / (511/2)    # normalization to range: -1 to 1
        
        x1 = x0
        x1, w1 = self.shape_self_attn[self.arg['n_layers'] + 1](x1, template)
        x1, w1 = self.shape_self_attn[self.arg['n_layers']](x1, template)
    
        y_list, w_y_list, yp = self.image_self_attn(feature128)

        x2 = x1
        for n in range(self.arg['n_layers'] - 1, 0, -1):
            x2, w2 = self.shape_to_image_attn[n](x2, xp, y_list[n], yp)
            x2, w2 = self.shape_self_attn[n](x2, template)#use template, not xp
            # x2 (B,N,E)
        
        # Refinement Blocks
        u0 = self.out(x2) #(B,N,2)      
        #------------------
        obj0 = obj_init+u0
        #------------------
        x3 = self.shape_embedding1(obj0.detach(), template, feature512, 1)#(B,N,E), difference from T2
        for n in range(0, self.arg['n_layers']):            
            x3, w3 = self.shape_self_attn1[n](x3, template)#use template, not xp
        u1 = 2 * self.out1(x3)    
        obj1 = obj0 + u1
        #------------------
        return obj0, obj1