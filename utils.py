import numpy as np
import torch

def preprocess_convert_images(images):
    images = images.astype(np.float32)
    images = (images - 127.5) / 127.5
    images = np.transpose(images, (0,3,1,2)) 
    return torch.from_numpy(images)

def postprocess_convert_images(images):
    images = images.numpy()
    images = images * 127.5 + 127.5
    images = images.astype(np.uint8)
    images = np.transpose(images, (0,2,3,1)) 
    return images

def crossover_mean(z_female, z_male, child_index):
    if child_index == 0: # 100% female.
        return z_female
    if child_index == 4: # 100% male.
        return z_male
    return z_female * (1.0 - child_index * 0.25) + z_male * child_index * 0.25

def crossover_split(z_female, z_male, child_index, device):
    z_dim = z_female.shape
    if child_index == 0: # 100% female.
        return z_female
    elif child_index == 1: # 75/25 F/M.
        mask = np.array([int(i%4!=0) for i in xrange(z_dim[1] * z_dim[2] * z_dim[3])]).reshape((z_dim[1], z_dim[2], z_dim[3]))
        mask = torch.from_numpy(np.array([mask for _ in xrange(z_dim[0])])).to(device, torch.float)
        return z_female * mask + z_male * (1 - mask)
    elif child_index == 2: # 50/50 F/M.
        mask = np.array([i%2 for i in xrange(z_dim[1] * z_dim[2] * z_dim[3])]).reshape((z_dim[1], z_dim[2], z_dim[3]))
        mask = torch.from_numpy(np.array([mask for _ in xrange(z_dim[0])])).to(device, torch.float)
        return z_female * mask + z_male * (1 - mask)
    elif child_index == 3: # 25/75 F/M.
        mask = np.array([int(i%4==0) for i in xrange(z_dim[1] * z_dim[2] * z_dim[3])]).reshape((z_dim[1], z_dim[2], z_dim[3]))
        mask = torch.from_numpy(np.array([mask for _ in xrange(z_dim[0])])).to(device, torch.float)
        return z_female * mask + z_male * (1 - mask)
    elif child_index == 4: # 100% male.
        return z_male

def crossover_part(z_female, z_male, child_index, device):
    z_dim = z_female.shape
    flag = np.arange(z_dim[1] * z_dim[2] * z_dim[3])
    if child_index == 0: # 100% female.
        return z_female
    elif child_index == 1: # 75/25 F/M.
        mask = (flag < z_dim[1] * z_dim[2] * z_dim[3] * 3 / 4).astype(np.int).reshape((z_dim[1], z_dim[2], z_dim[3]))
        mask = torch.from_numpy(np.array([mask for _ in xrange(z_dim[0])])).to(device, torch.float)
        return z_female * mask + z_male * (1 - mask)
    elif child_index == 2: # 50/50 F/M.
        mask = (flag < z_dim[1] * z_dim[2] * z_dim[3] / 2).astype(np.int).reshape((z_dim[1], z_dim[2], z_dim[3]))
        mask = torch.from_numpy(np.array([mask for _ in xrange(z_dim[0])])).to(device, torch.float)
        return z_female * mask + z_male * (1 - mask)
    elif child_index == 3: # 25/75 F/M.
        mask = (flag < z_dim[1] * z_dim[2] * z_dim[3] / 4).astype(np.int).reshape((z_dim[1], z_dim[2], z_dim[3]))
        mask = torch.from_numpy(np.array([mask for _ in xrange(z_dim[0])])).to(device, torch.float)
        return z_female * mask + z_male * (1 - mask)
    if child_index == 4: # 100% male.
        return z_male

def crossover(z_female, z_male, child_index, mode, device): # mode: "mean", "split", "part".
    assert child_index >= 0 and child_index <= 4
    assert mode in ["mean", "split", "part"]
    if mode == "mean":
        return crossover_mean(z_female, z_male, child_index)
    elif mode == "split":
        return crossover_split(z_female, z_male, child_index, device)
    elif mode == "part":
        return crossover_part(z_female, z_male, child_index, device)

