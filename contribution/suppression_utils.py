
import numpy as np
import torch
import cv2
import random


def tensor_to_numpy(img_tensor):
    return img_tensor.permute(1, 2, 0).numpy().astype(np.float32)


def numpy_to_tensor(img_np):
    return torch.from_numpy(img_np).permute(2, 0, 1)


def patch_shuffle_tensor(img_tensor, grid_size=6):
    img = tensor_to_numpy(img_tensor)
    H, W, C = img.shape
    ph, pw = H // grid_size, W // grid_size
    patches = [
        img[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :].copy()
        for i in range(grid_size) for j in range(grid_size)
    ]
    random.shuffle(patches)
    result = np.zeros_like(img)
    for idx, (i, j) in enumerate(
        [(i, j) for i in range(grid_size) for j in range(grid_size)]
    ):
        result[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :] = patches[idx]
    return numpy_to_tensor(result)


def bilateral_filter_tensor(img_tensor, d=11,
                             sigma_color=170, sigma_space=75):
    img = tensor_to_numpy(img_tensor)
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)
    return numpy_to_tensor(filtered.astype(np.float32) / 255.0)


def grayscale_tensor(img_tensor):
    img = tensor_to_numpy(img_tensor)
    gray = (0.299 * img[:,:,0] +
            0.587 * img[:,:,1] +
            0.114 * img[:,:,2])
    gray_3ch = np.stack([gray, gray, gray], axis=2).astype(np.float32)
    return numpy_to_tensor(gray_3ch)


def apply_suppression(img_tensor, mode, grid_size=6):
    if mode == 'normal':
        return img_tensor
    elif mode == 'shape':
        return patch_shuffle_tensor(img_tensor, grid_size)
    elif mode == 'texture':
        return bilateral_filter_tensor(img_tensor)
    elif mode == 'color':
        return grayscale_tensor(img_tensor)
    else:
        raise ValueError(f"Unknown mode: {mode}")
