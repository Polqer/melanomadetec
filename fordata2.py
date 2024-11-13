import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import os
import copy
import time
import gc
from google.colab import drive

from step_1 import *

# Пути к данным на Google Диске
second_data_dir = '/content/drive/MyDrive/Dataset2'  # Путь ко второму набору данных
fine_tuned_result_dir = '/content/drive/MyDrive/FineTuningResults'
batch_size = 16
num_epochs = 30  # Сокращаем количество эпох для fine-tuning
learning_rate = 0.00005

# Преобразования для второго набора данных
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Подготовка второго набора данных
def prepare_datasets(data_dir, val_split=0.15, test_split=0.15):
    dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
    train_size = int((1 - val_split - test_split) * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    # Устанавливаем соответствующие преобразования для валидационной и тестовой выборок
    val_dataset.dataset.transform = data_transforms['val']
    test_dataset.dataset.transform = data_transforms['val']
    
    return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = prepare_datasets(second_data_dir)
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
class_names = train_dataset.dataset.classes
class_num = len(class_names)