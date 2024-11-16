import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import random_split, DataLoader, Subset
import time
import os
import copy
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import ImageFile
from sklearn.model_selection import train_test_split
from google.colab import drive
import gc
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark=True
gc.collect()
drive.mount('/content/drive')
# Пути к данным на Google Диске
data_dir = '/content/drive/MyDrive/Dataset1'
result_dir = '/content/drive/MyDrive/Results'
torch.cuda.empty_cache()

batch_sz = 16
num_epoch = 20
init_learning_rate = 0.0001
learning_rate_decay_factor = 0.2
num_epochs_decay = 15
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),}

def prepare_datasets(data_dir, train_split=0.7, val_split=0.15, test_split=0.15):
    # Загружаем полный датасет
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])

    # Делим на тренировочную и оставшуюся выборку (валидация + тест)
    train_size = int(train_split * len(full_dataset))
    remaining_size = len(full_dataset) - train_size

    # Оставшуюся выборку делим на валидацию и тест
    val_size = int(val_split * remaining_size)
    test_size = remaining_size - val_size

    # Разделение на три части
    train_dataset, remaining_dataset = random_split(full_dataset, [train_size, remaining_size])
    val_dataset, test_dataset = random_split(remaining_dataset, [val_size, test_size])

    # Применяем соответствующие трансформации
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']
    test_dataset.dataset.transform = data_transforms['val']

    return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = prepare_datasets(data_dir)

# Создаём загрузчики данных
dataloaders = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=2),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_sz, shuffle=True, num_workers=2),
    'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_sz, shuffle=False, num_workers=2)
}

dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset),
    'test': len(test_dataset)
}
class_names = train_dataset.dataset.classes
class_num = len(class_names)
#исходный объект ImageFolder
full_dataset = val_dataset.dataset.dataset
# индексы классов в val_dataset
val_labels = [val_dataset[i][1] for i in range(len(val_dataset))]
#считаем, сколько примеров каждого класса
val_class_counts = Counter(val_labels)
#количество примеров каждого класса
print("Validation dataset class counts:")
for cls_idx, count in val_class_counts.items():
    print(f"Class {full_dataset.classes[cls_idx]}: {count}")
#получаем индексы классов в train_dataset
train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
train_class_counts = Counter(train_labels)
print("\nTraining dataset class counts:")
for cls_idx1, count in train_class_counts.items():
    print(f"Class {full_dataset.classes[cls_idx1]}: {count}")


print(f"Train size: {dataset_sizes['train']}, Validation size: {dataset_sizes['val']}, Test size: {dataset_sizes['test']}")