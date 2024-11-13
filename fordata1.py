import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import random_split, DataLoader
import time
import os
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, confusion_matrix, roc_auc_score
from PIL import ImageFile
from google.colab import drive
import gc
# Пути к данным на Google Диске
data_dir = '/content/drive/MyDrive/Dataset1'
result_dir = '/content/drive/MyDrive/Results'
# Пропорции для разделения
train_split = 0.7  # 70% данных на обучение
val_split = 0.15   # 15% данных на валидацию
test_split = 0.15  # 15% данных на тестирование
batch_sz = 16

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

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
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}
# Загрузка данных
dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
# Определение размеров подвыборок
train_size = int(train_split * len(dataset))
val_size = int(val_split * len(dataset))
test_size = len(dataset) - train_size - val_size
#Разделение на обучающую, валидационную и тестовую выборки
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
val_dataset.dataset.transform = data_transforms['val']
test_dataset.dataset.transform = data_transforms['test']
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=2),
    'val': DataLoader(val_dataset, batch_size=batch_sz, shuffle=True, num_workers=2),
    'test': DataLoader(test_dataset, batch_size=batch_sz, shuffle=False, num_workers=2)  # test set should not be shuffled
}
# Размеры датасетов(?)запуск отдельно
dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset),
    'test': len(test_dataset)
}
class_names = dataset.classes
class_num = len(class_names)
print(f"Train size: {dataset_sizes['train']}, Validation size: {dataset_sizes['val']}, Test size: {dataset_sizes['test']}")