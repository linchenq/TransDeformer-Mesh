import numpy as np
import re
from scipy.ndimage import shift as scipy_shift_function
import torch

def data_shift(image, label, obj, shift_yx):
    shift_ymax, shift_xmax = shift_yx
    shift_x = np.random.randint(-shift_xmax, shift_xmax + 1)
    shift_y = np.random.randint(-shift_ymax, shift_ymax + 1)
    
    shift_image = scipy_shift_function(input=image, shift=(shift_y, shift_x), order=0)
    shift_label = scipy_shift_function(input=label, shift=(shift_y, shift_x), order=0)

    shift_obj = np.zeros_like(obj)
    shift_obj[:, 0] = obj[:, 0] + shift_x
    shift_obj[:, 1] = obj[:, 1] + shift_y

    return shift_image, shift_label, shift_obj

'''
dict = {'image': (H, W), ndarray
        'shape': (N, 2), ndarray,       # landmark: [:24]
        'shape_sampled_39_19': (M, 2), ndarray
        'label': (H, W), ndarray
        }
'''

class AugmentSpineDataset(torch.utils.data.Dataset):
    def __init__(self, pth, num_classes, max_index, transform, shift_yx, random_seed):
        self.transform = transform
        self.num_classes = num_classes
        self.max_index = max_index
        self.shift_yx = [int(shift_yx[0]), int(shift_yx[1])]
        self.rng=np.random.RandomState(random_seed)
        with open(pth, "r") as f:
            self.files = f.readlines()
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        idx_A = self.rng.randint(0, self.max_index + 1)
        idx_B = self.rng.randint(0, self.max_index + 1)
        
        pth_shared = self.files[index % len(self.files)].rstrip()
        file_name = re.split(r'[\\/]', pth_shared)[-2]
        file_idx = re.split(r'[\\/]', pth_shared)[-1]
        
        pth_A = pth_shared + "_i{}.pt".format(idx_A)
        pth_B = pth_shared + "_i{}.pt".format(idx_B)
        data_A = torch.load(pth_A)
        data_B = torch.load(pth_B)
        
        image_A = data_A['image']
        label_A = data_A['label']
        obj_A = data_A['shape_sampled_39_19']
        obj_B = data_B['shape_sampled_39_19']
        
        if self.num_classes == 2:
            label_A[label_A>0]=1    
        
        if self.transform:
            image_A, label_A, obj_A = data_shift(image=image_A,
                                                 label=label_A,
                                                 obj=obj_A,
                                                 shift_yx=self.shift_yx)
        
        # segmentation
        # num_classes=12: {0: background; 1 ~ 6: bones from top to bottom; 7 ~ 11: disks from top to bottom}
        # num_classes=2: 0 background, 1 other
        mask_A = np.zeros((self.num_classes, label_A.shape[0], label_A.shape[1]), dtype=np.float32)
        for idx in range(self.num_classes):
            mask_A[idx] = (label_A == idx)
        
        image_A = image_A.reshape(1, image_A.shape[0], image_A.shape[1])
        
        # numpy to tensor
        image_A = torch.tensor(image_A, dtype=torch.float32)  # (1, H, W)
        label_A = torch.tensor(label_A, dtype=torch.int64)    # (H, W)
        mask_A = torch.tensor(mask_A, dtype=torch.float32)    # (num_classes, H, W)
        obj_A = torch.tensor(obj_A, dtype=torch.float32)      # (N, 2)
        obj_B = torch.tensor(obj_B, dtype=torch.float32)      # (N, 2)

        return image_A, label_A, mask_A, obj_A, obj_B, (idx_A, idx_B, file_name, file_idx)

#%%
if __name__ == '__main__':
    
    DEBUG_DATALOADER = True
    DEBUG_SHIFT = True
    
    import matplotlib.pyplot as plt
    dataset = AugmentSpineDataset(pth='./txt/train.txt',
                                  num_classes=12,
                                  max_index=20,
                                  transform=True,
                                  shift_yx=[16, 16],
                                  random_seed=0)
    print(f"len function: Dataset Length is {len(dataset)}")
    
    if DEBUG_DATALOADER:
        for index, (imgA, labelA, maskA, objA, objB, pairs) in enumerate(dataset):
            # Check objA
            fig, ax = plt.subplots()
            ax.imshow(imgA[0], cmap='gray')
            ax.plot(objA[:, 0], objA[:, 1], '.', markersize=0.3)
            ax.plot(objA[:24, 0], objA[:24, 1], 'ro', markersize=3)
    
            # Check pairs
            fig, ax = plt.subplots()
            ax.imshow(imgA[0], cmap='gray')
            ax.plot(objB[:, 0], objB[:, 1], 'c.', markersize=0.25)
            ax.plot(objA[:, 0], objA[:, 1], 'r.', markersize=0.25)
            
            # Check mask
            for j, mask in enumerate(maskA):
                fig, ax = plt.subplots()
                assert (mask == maskA[j]).all()
                ax.imshow(imgA[0] * 0.5 + mask, cmap='gray')
                ax.plot(objA[:24, 0], objA[:24, 1], 'ro', markersize=3)
                
            if index >= 1:
                break
    
    if DEBUG_SHIFT:
        def unit_test_shift(xmax, ymax, title):
            dataset = AugmentSpineDataset(pth='./txt/train.txt',
                                          num_classes=12,
                                          max_index=20,
                                          transform=True,
                                          shift_yx=[ymax, xmax],
                                          random_seed=0)
            imgA, labelA, maskA, objA, _, _ = dataset[0]
            fig, ax = plt.subplots()
            ax.imshow(imgA[0], cmap='gray')
            ax.plot(objA[:, 0], objA[:, 1], '.', markersize=0.3)
            ax.plot(objA[:24, 0], objA[:24, 1], 'ro', markersize=3)
            ax.set_title(title)
       
        unit_test_shift(xmax=0, ymax=0, title="Original")
        unit_test_shift(xmax=100, ymax=0, title="X-axis movement")
        
        unit_test_shift(xmax=0, ymax=0, title="Original")
        unit_test_shift(xmax=0, ymax=100, title="Y-axis movement")
       
            
        
        
        
        
        
        
        
        
        