
import sys, os, argparse, json
sys.path.append('/kaggle/working/contribution')

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from curriculum_dataset import CurriculumDataset, get_curriculum_probs
from tqdm import tqdm

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD  = [0.2023, 0.1994, 0.2010]


def get_loaders(mode='standard', batch_size=128,
                data_root='/kaggle/working/data'):
    normalize = T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)

    val_transform = T.Compose([T.Resize((32, 32)), T.ToTensor(), normalize])
    val_set = CIFAR10(root=data_root, train=False, download=True,
                      transform=val_transform)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    if mode == 'standard':
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
            normalize,
        ])
        train_set = CIFAR10(root=data_root, train=True, download=True,
                            transform=train_transform)
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)
        return train_loader, val_loader, None

    elif mode == 'curriculum':
        base_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
        ])
        raw_train = CIFAR10(root=data_root, train=True, download=True,
                            transform=base_transform)
        curriculum_set = CurriculumDataset(
            base_dataset=raw_train,
            normalize=normalize,
        )
        train_loader = DataLoader(curriculum_set, batch_size=batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)
        return train_loader, val_loader, curriculum_set


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)
            correct += model(imgs).argmax(1).eq(labels).sum().item()
            total   += labels.size(0)
    return correct / total


def train(mode='standard', epochs=30, batch_size=128, save_path=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'═' * 80}")
    print(f"{' ' * 25}TRAINING STARTED - {mode.upper()} MODE")
    print(f"{'═' * 80}")
    print(f"  Epochs      : {epochs}")
    print(f"  Batch size  : {batch_size}")
    print(f"  Device      : {device}")
    
    if mode == 'curriculum':
        p0 = get_curriculum_probs(1, epochs)
        pT = get_curriculum_probs(epochs, epochs)
        print(f"  Curriculum  : Starts → normal={p0['normal']:.0%} | "
              f"shape={p0['shape']:.0%} | texture={p0['texture']:.0%} | color={p0['color']:.0%}")
        print(f"                Ends   → normal={pT['normal']:.0%} | "
              f"shape={pT['shape']:.0%} | texture={pT['texture']:.0%} | color={pT['color']:.0%}")
    
    print(f"{'═' * 80}\n")

    model = models.resnet18(weights=None, num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader, curriculum_set = get_loaders(mode, batch_size)
    history = []

    for epoch in range(1, epochs + 1):
        # Update curriculum schedule
        if mode == 'curriculum' and curriculum_set is not None:
            curriculum_set.set_epoch(epoch, epochs)
            probs = get_curriculum_probs(epoch, epochs)

        model.train()
        correct = total = 0
        running_loss = 0.0
        mode_counts = {'normal': 0, 'shape': 0, 'texture': 0, 'color': 0}

        print(f"\n Epoch {epoch:02d}/{epochs} {'─' * 60}")

        pbar = tqdm(train_loader, desc="  Training", 
                    leave=True, dynamic_ncols=True, ncols=100)

        for batch in pbar:
            imgs = batch[0].to(device)
            labels = batch[1].to(device)
            
            if len(batch) == 3 and mode == 'curriculum':
                for m in batch[2]:
                    mode_counts[m] += 1

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            correct += out.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.4f}',
                'acc': f'{correct / total:.4f}'
            })

        scheduler.step()

        train_acc = correct / total
        avg_loss = running_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)

        history.append({
            'epoch': epoch,
            'train_loss': round(avg_loss, 4),
            'train_acc': round(train_acc, 4),
            'val_acc': round(val_acc, 4),
        })

        # Final epoch summary
        print(f"  → Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if mode == 'curriculum':
            total_s = sum(mode_counts.values())
            if total_s > 0:
                actual = " | ".join(f"{k}={v/total_s:.1%}" for k, v in mode_counts.items())
                scheduled = " | ".join(f"{k}={probs[k]:.1%}" for k in mode_counts)
                print(f"  → Actual distribution    : {actual}")
                print(f"  → Scheduled distribution : {scheduled}")

        print(f"{'─' * 80}")

    # Save model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'mode': mode,
        }, save_path)
        print(f"\n Model successfully saved → {save_path}")

    print(f"\n Training completed successfully: {mode.upper()} mode\n")
    return model, history
