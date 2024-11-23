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
#пути
data_dir = '/content/drive/MyDrive/dataset_split'
result_dir = '/content/drive/MyDrive/Results'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')
torch.cuda.empty_cache()

torch.backends.cuda.matmul.allow_tf32 = True


batch_sz = 8
num_epoch = 20
init_learning_rate = 0.0001
learning_rate_decay_factor = 0.2
num_epochs_decay = 15
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# data_transforms = {'train': transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.Resize(256),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)]),
#     'val': transforms.Compose([
#         transforms.Resize(256),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)]),}

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# image_datasets = {'train': datasets.ImageFolder(train_dir, data_transforms['train']),
#                               'val': datasets.ImageFolder(test_dir, data_transforms['val'])}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sz,
#                                               shuffle=True, num_workers=4)
#                for x in ['train', 'val']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = image_datasets['train'].classes
# class_num = len(class_names)

#данные из папок

train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])

dataloaders = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_sz, shuffle=True)
}



dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset)}
class_names = train_dataset.classes
class_num = len(class_names)
# #кол-во изображений для каждого класса
# def count_class_samples(dataset, dataset_name):
#     class_counts = Counter([sample[1] for sample in dataset])
#     print(f"\n{dataset_name} dataset class counts:")
#     for cls_idx, count in class_counts.items():
#         print(f"Class {dataset.classes[cls_idx]}: {count}")


# count_class_samples(train_dataset, "train")

# count_class_samples(val_dataset, "validation")



# #Ссылаемся на исходный объект ImageFolder
# full_dataset = val_dataset.dataset.dataset

# #Получаем индексы классов в val_dataset
# val_labels = [val_dataset[i][1] for i in range(len(val_dataset))]
# #Считаем, сколько примеров каждого класса
# val_class_counts = Counter(val_labels)
# # Выводим количество примеров каждого класса
# print("Validation dataset class counts:")
# for cls_idx, count in val_class_counts.items():
#     print(f"Class {full_dataset.classes[cls_idx]}: {count}")


# #Получаем индексы классов в train_dataset
# train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
# #Считаем, сколько примеров каждого класса
# train_class_counts = Counter(train_labels)
# #Выводим количество примеров каждого класса
# print("\nTraining dataset class counts:")
# for cls_idx1, count in train_class_counts.items():
#     print(f"Class {full_dataset.classes[cls_idx1]}: {count}")


#print(f"Train size: {dataset_sizes['train']}, Validation size: {dataset_sizes['val']}, Test size: {dataset_sizes['test']}")