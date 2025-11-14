# src/data/fashion_mnist.py

from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_transforms(img_size=224):
    """
    Return torchvision transforms for train/val/test.
    Train: resize -> random horizontal flip -> to tensor -> normalize
    Val/Test: resize -> to tensor -> normalize
    """
    # TODO: implement with torchvision.transforms
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    to_3ch = transforms.Grayscale(num_output_channels=3)

    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),        # makes all images the same size
        transforms.RandomHorizontalFlip(p=0.5),         # small randomness -> better generlization
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        normalize,                                      # center around 0, scale to ~[-1,1]
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tfms, eval_tfms



def get_dataloaders(data_root="data/raw", batch_size=64, num_workers=2, img_size=224, val_split=0.1):
   """
    Download Fashion-MNIST to data_root, create train/val/test datasets,
    apply transforms, and return three DataLoaders:
    (train_loader, val_loader, test_loader)
    - Keep shuffle=True for train, False for val/test
    - Use torch.utils.data.random_split for the val split
    """
   data_root = Path(data_root)
   data_root.mkdir(parents=True, exist_ok=True)

   train_tfms, eval_tfms = get_transforms(img_size=img_size)

   # Full training set (split later into train/val)
   train_full = datasets.FashionMNIST(
       root =str(data_root), train=True, download=True, transform=train_tfms
   )
   # Held-out test set 
   test_ds = datasets.FashionMNIST(
       root=str(data_root), train=False, download=True, transform=eval_tfms
   )

   # Split into train /val
   val_size = int(len(train_full) * val_split)
   train_size = len(train_full) - val_size
   train_ds, val_ds = random_split(train_full, [train_size, val_size])

   # IMPORTANT: validation shouldnt use random augmentations
   val_ds.dataset.transform = eval_tfms

   # DataLoaders control batching/shuffling/parallel reads
   train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
   val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
   test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
   return train_loader, val_loader, test_loader

    # TODO: implement with torchvision.datasets.FashionMNIST + DataLoader
