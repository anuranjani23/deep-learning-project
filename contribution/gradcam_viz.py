
import torch
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR10
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD  = [0.2023, 0.1994, 0.2010]
CLASSES    = ['airplane','automobile','bird','cat','deer',
              'dog','frog','horse','ship','truck']
device     = torch.device('cuda')


def load_model(path):
    m = models.resnet18(weights=None, num_classes=10).to(device)
    ckpt = torch.load(path, map_location=device)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return m


def get_cam_grid(std_model, cur_model, dataset,
                 class_idx=5, n_images=8):
    """
    For n_images from class_idx, show:
      col 0 — original image
      col 1 — standard model Grad-CAM
      col 2 — curriculum model Grad-CAM
    """
    normalize  = T.Normalize(CIFAR_MEAN, CIFAR_STD)
    std_cam    = GradCAM(model=std_model,
                         target_layers=[std_model.layer4[-1]])
    cur_cam    = GradCAM(model=cur_model,
                         target_layers=[cur_model.layer4[-1]])

    # collect images for this class
    samples = []
    for img, label in dataset:
        if label == class_idx:
            samples.append(img)
        if len(samples) == n_images:
            break

    fig, axes = plt.subplots(n_images, 3,
                             figsize=(6, n_images * 2.2))
    fig.suptitle(
        f'Grad-CAM: class "{CLASSES[class_idx]}" — '
        f'standard vs curriculum',
        fontsize=11, y=1.01
    )

    col_titles = ['Original', 'Standard', 'Curriculum']
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=10)

    for row, img_tensor in enumerate(samples):
        # raw numpy for overlay (H×W×3, float32, 0-1)
        img_np = img_tensor.permute(1, 2, 0).numpy().astype(np.float32)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        # input tensor for Grad-CAM
        inp = normalize(img_tensor).unsqueeze(0).to(device)

        std_mask = std_cam(input_tensor=inp)
        cur_mask = cur_cam(input_tensor=inp)

        std_viz = show_cam_on_image(img_np, std_mask[0], use_rgb=True)
        cur_viz = show_cam_on_image(img_np, cur_mask[0], use_rgb=True)

        # upscale 32px → 128px for visibility
        def up(arr):
            return cv2.resize(arr, (128, 128),
                              interpolation=cv2.INTER_NEAREST)

        axes[row][0].imshow(up((img_np * 255).astype(np.uint8)))
        axes[row][1].imshow(up(std_viz))
        axes[row][2].imshow(up(cur_viz))

        for col in range(3):
            axes[row][col].axis('off')

    plt.tight_layout()
    out = ('/kaggle/working/contribution/'
           f'gradcam_{CLASSES[class_idx]}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {out}")


if __name__ == '__main__':
    raw_val = CIFAR10(
        root='/kaggle/working/data', train=False, download=True,
        transform=T.Compose([T.Resize((32, 32)), T.ToTensor()])
    )
    std_model = load_model(
        '/kaggle/working/contribution/models/standard.pt')
    cur_model = load_model(
        '/kaggle/working/contribution/models/curriculum.pt')

    # run for a few visually distinct classes
    for cls in [5, 1, 8]:   # dog, automobile, ship
        get_cam_grid(std_model, cur_model, raw_val,
                     class_idx=cls, n_images=8)
