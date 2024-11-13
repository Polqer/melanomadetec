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

from fordata2 import *

# Загрузка предварительно обученной модели
model_ft = EnsembleModel(class_num)
checkpoint_path = '/content/drive/MyDrive/Results/best_ensemble_model.ckpt'
model_ft.load_state_dict(torch.load(checkpoint_path))
model_ft = model_ft.to(device)

# Настройка функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

# Обучение модели на втором наборе данных с подсчетом метрик
def train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_labels = []
            all_preds = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)

            # Вычисление метрик
            epoch_acc = accuracy_score(all_labels, all_preds)
            epoch_precision = precision_score(all_labels, all_preds, average='weighted')
            epoch_recall = recall_score(all_labels, all_preds, average='weighted')
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
            epoch_auc = roc_auc_score(all_labels, all_preds, average='weighted', multi_class='ovr')

            if phase == 'train':
                train_accuracies.append(epoch_acc)
            else:
                val_accuracies.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f} Prec: {:.4f} Recall: {:.4f} F1: {:.4f} AUC: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_auc))

            # Сохранение лучшей модели
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # Очистка кеша
        torch.cuda.empty_cache()
        gc.collect()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)

    # Построение графиков потерь и точности
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.tight_layout()
    plt.show()

    return model

# Запуск fine-tuning
fine_tuned_model = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)

# Сохранение fine-tuned модели
os.makedirs(fine_tuned_result_dir, exist_ok=True)
torch.save(fine_tuned_model.state_dict(), os.path.join(fine_tuned_result_dir, 'ensemble_model_2.ckpt'))
