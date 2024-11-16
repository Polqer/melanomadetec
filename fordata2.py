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



import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
drive.mount('/content/drive')
# Пути к данным на Google Диске
second_data_dir = '/content/drive/MyDrive/Dataset2'  # Путь ко второму набору данных
fine_tuned_result_dir = '/content/drive/MyDrive/FineTuningResults'



import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Пути к данным на Google Диске
second_data_dir = '/content/drive/MyDrive/Dataset2'  # Путь ко второму набору данных
fine_tuned_result_dir = '/content/drive/MyDrive/FineTuningResults'

# Гиперпараметры
batch_size = 16
num_epochs = 2  # Количество эпох для fine-tuning
learning_rate = 0.00005

# Преобразования для данных
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

# Подготовка данных
def prepare_datasets(data_dir):
    # Загружаем данные из поддиректорий train и val
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
    
    return train_dataset, val_dataset

# Загрузка данных
train_dataset, val_dataset = prepare_datasets(second_data_dir)

# Создаем DataLoader для работы с данными
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2),
}

# Основные параметры данных
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = train_dataset.classes  # Названия классов
class_num = len(class_names)  # Количество классов

# Вывод информации
print(f"Классы: {class_names}")
print(f"Размеры датасетов: {dataset_sizes}")


