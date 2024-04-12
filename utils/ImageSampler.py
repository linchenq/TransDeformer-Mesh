import torch
# %%
def SampleImage2D(image, point, origin=None, mode='bilinear', padding_mode='zeros', align_corners=True,
                  output_channel_last=True):
    # image.shape:  B x C x H x W (or 1 x C x H x W),  H is y-axis, W is x-axis
    # point.shape:  B x N x 2 (or 1 x N x 2)
    # point[0,0,:] is (x,y): if it is (y,x), then use torch.flip(point, dims=[2]) before call this function
    # origin: image[:,:,0,0] in point space, origin=[x0, y0]
    #    it could be [0.5, 0.5] in matlab and sk-image-poly2mask
    #    it could be [0, 0] for medical image
    # assume pixel spacing is [1,1] in image
    
    if len(image.shape) != 4:
        raise ValueError("len(image.shape) != 4")
    if len(point.shape) != 3:
        raise ValueError("len(point.shape) != 3")
    
    if origin is None:
        origin = [0, 0]
    if image.shape[0] == 1 and point.shape[0] > 1:
        image = image.expand(point.shape[0], image.shape[1], image.shape[2], image.shape[3])
    elif point.shape[0] == 1 and image.shape[0] > 1:
        point = point.expand(image.shape[0], point.shape[1], 2)
    elif point.shape[0] != image.shape[0]:
        raise ValueError("point.shape and image.shape do not match")
    H = image.shape[2]
    W = image.shape[3]
    x = point[: , :, 0:1]
    y = point[:, :, 1:2]
    x = -1 + (x - origin[0]) * 2 / (W - 1)
    y= -1 +(y - origin[1]) * 2 / (H - 1)
    point = torch.cat([x, y], dim=2)
    # print('1', point.shape)
    # mesh.shape:  B x N x 2 => B x N x 1 x 2 to use grid_sample
    point = point.view(point.shape[0], point.shape[1], 1, 2)
    # print('2', point.shape)
    output = torch.nn.functional.grid_sample(input=image, grid=point,
                                             mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    # print('3', output.shape)
    # output.shape: B x C x N x 1 => B x C x N
    output = output.view(output.shape[0], output.shape[1], output.shape[2])
    # print('4', output.shape)   
    if output_channel_last == True:
        # output.shape: B x C x N => B x N x C
        output=output.permute(0, 2, 1)
    # print('5', output.shape)
    return output
# %%
def SampleImage3D(image, point, origin=None, mode='bilinear', padding_mode='zeros', align_corners=True,
                  output_channel_last=True):
    # image.shape:  B x C x D x H x W (or 1 x C x D x H x W)
    #               D is z-axis, H is y-axis, W is x-axis
    # point.shape:  B x N x 3 (or 1 x N x 3)
    # point[0,0,:] is (x,y,z)
    # origin: image[:,:,0,0,0] in point space, origin=[x0, y0, z0]
    #         it could be [0, 0, 0] for medical image
    # assume pixel spacing is [1,1,1] in image
    
    if len(image.shape) != 5:
        raise ValueError("len(image.shape) != 5")
    if len(point.shape) != 3:
        raise ValueError("len(point.shape) != 3")
        
    if origin is None:
        origin = [0, 0, 0]
    if image.shape[0] == 1 and point.shape[0] > 1:
        image = image.expand(point.shape[0], image.shape[1], image.shape[2], image.shape[3], image.shape[4])
    elif point.shape[0]==1 and image.shape[0] > 1:
        point = point.expand(image.shape[0], point.shape[1], 3)
    elif point.shape[0] != image.shape[0]:
        raise ValueError("point.shape and image.shape do not match")
    D = image.shape[2]
    H = image.shape[3]
    W = image.shape[4]
    x = point[:,:,0:1]
    y = point[:,:,1:2]
    z = point[:,:,2:3]
    x = -1 + (x - origin[0]) * 2 / (W - 1)
    y = -1 + (y - origin[1]) * 2 / (H - 1)
    z = -1 + (z - origin[2]) * 2/ (D - 1)
    point = torch.cat([x, y, z], dim=2)
    # print('1', point.shape)
    # mesh.shape:  B x N x 3 => B x N x 1 x 1 x 3 to use grid_sample
    point = point.view(point.shape[0], point.shape[1], 1, 1, 3)
    # print('2', point.shape)
    output = torch.nn.functional.grid_sample(input=image, grid=point,
                                             mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    # print('3', output.shape)
    # output.shape: B x C x N x 1 x 1 x 1 => batch x C x N
    output = output.view(output.shape[0], output.shape[1], output.shape[2])
    # print('4', output.shape)
    if output_channel_last == True:
        # output.shape: B x C x N => B x N x C
        output=output.permute(0, 2, 1)
    # print('5', output.shape)
    return output
# %%
if __name__ == '__main__':
    image = torch.zeros((1, 1, 10, 10), dtype=torch.float32)
    image[0, :, 3, 9]=1
    point = torch.zeros((1, 1, 2), dtype=torch.float32)
    point[0, 0, 0] = 9
    point[0, 0, 1] = 3
    output = SampleImage2D(image, point, origin=[0, 0], mode='nearest', padding_mode='zeros', align_corners=True)
    print(output)
    output = SampleImage2D(image, point, origin=[0, 0], mode='nearest', padding_mode='zeros', align_corners=False)
    print(output)
    # %%
    image = torch.zeros((1, 6, 10, 10, 10), dtype=torch.float32)
    image[0, :, 3, 6, 9] = 1
    point = torch.zeros((1, 1, 3), dtype=torch.float32)
    point[0, 0, 0] = 9
    point[0, 0, 1] = 6
    point[0, 0, 2] = 3
    output = SampleImage3D(image, point, origin=[0,0,0], mode='nearest', padding_mode='zeros', align_corners=True)
    print(output.shape)
    print(output)
    output = SampleImage3D(image, point, origin=[0,0,0], mode='nearest', padding_mode='zeros', align_corners=False)
    print(output.shape)
    print(output)

