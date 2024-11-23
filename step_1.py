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
from sklearn.preprocessing import label_binarize
import wandb
from fordata1 import *

wandb.init(project="my-awesome-project", name="step1_checki_2")
#cоздаем ансамблевую модель, ResNet152 + ViT
class EnsembleModel(nn.Module):
    def __init__(self, class_num):
        super(EnsembleModel, self).__init__()
        #ResNet152
        self.resnet = models.resnet152()  # Используем ResNet152
        num_ftrs_resnet = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # - последний слой, чтобы использовать его выходы
        #ViT
        self.vit = models.vit_b_16()
        num_ftrs_vit = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Identity()  # аналогично
        # Общий классификационный слой для объединенных признаков
        self.classifier = nn.Linear(num_ftrs_resnet + num_ftrs_vit, class_num)

    def forward(self, x):
        #прямой проход через ResNet и ViT
        resnet_out = self.resnet(x)
        vit_out = self.vit(x)
        #сцепление выходов
        combined = torch.cat((resnet_out, vit_out), dim=1)
        #итог предсказание
        out = self.classifier(combined)
        return out

#функция для сохранения чекпоинта
def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir="/content/drive/MyDrive/Results"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.ckpt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

#функция для загрузки чекпоинта
def load_checkpoint(model, optimizer, scheduler, checkpoint_dir="/content/drive/MyDrive/Result"):
    checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    if not checkpoints:
        print("No checkpoint found, starting from scratch.")
        return model, optimizer, scheduler, 0  # Начать с нуля

    #найти последний чекпоинт по названию
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded: {latest_checkpoint}, resuming from epoch {start_epoch}")
    return model, optimizer, scheduler, start_epoch
def calculate_metrics(y_true, y_pred, y_true_proba, y_pred_proba):
    return {
        'accuracy': accuracy_score(y_true=y_true, y_pred=y_pred),
        'confusion_matrix': confusion_matrix(y_true=y_true, y_pred=y_pred),
        'micro/precision': precision_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0),
        'micro/recall': recall_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0),
        'micro/f1': f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0),
        'micro/roc_auc_score': roc_auc_score(y_true=y_true_proba, y_score=y_pred_proba, average='micro'),
        'macro/precision': precision_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0),
        'macro/recall': recall_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0),
        'macro/f1': f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0),
        'roc_auc_score': roc_auc_score(y_true=y_true_proba, y_score=y_pred_proba, average='samples'),
        'weighted/precision': precision_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0),
        'weighted/recall': recall_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0),
        'weighted/f1': f1_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0)
    }
#используем
checkpoint_dir = "/content/drive/MyDrive/Result"


#экземпляр ансамблевой модели
model_ft = EnsembleModel(class_num).to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=init_learning_rate)
#поменяла ReduceLROnPlateau
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=learning_rate_decay_factor, patience=10)

#загрузка чекпоинта, если он существует
model_ft, optimizer_ft, exp_lr_scheduler, start_epoch = load_checkpoint(model_ft, optimizer_ft, exp_lr_scheduler, checkpoint_dir)


def train_model(model, criterion, optimizer, scheduler, num_epochs=num_epoch, start_epoch=0):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    #хранение истории обучения
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            #массивы для подсчета метрик по классам
            all_labels = []
            all_preds = []
            all_true_proba = []
            all_pred_proba = []
            print(len(dataloaders[phase].dataset))  # Убедитесь, что есть данные
            print("Train set length:", len(dataloaders['train'].dataset))
            print("Validation set length:", len(dataloaders['val'].dataset))
            #обработка данных
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
#сохраняем вероятности для ROC-AUC
                    probabilities = torch.softmax(outputs, dim=1).detach().cpu().numpy()
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_true_proba.extend(np.eye(class_num)[labels.cpu().numpy()])
                all_pred_proba.extend(probabilities)
            #расчет метрик для эпохи
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            #сохранение значений потерь и точности для графиков
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc)






            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))



            if phase == 'val':
                #расчет дополнительных метрик
                metrics = calculate_metrics(
                    y_true=all_labels,
                    y_pred=all_preds,
                    y_true_proba=np.array(all_true_proba),
                    y_pred_proba=np.array(all_pred_proba)
                )
                print("\nValidation metrics:")
                for key, value in metrics.items():
                    if key != 'confusion_matrix':
                        print(f"{key}: {value:.4f}")


                #лоогирунм метрики в W&B
                wandb.log({
                    "Val Loss": epoch_loss,
                    "Val Accuracy": epoch_acc.item(),
                    **{f"Val {k}": v for k, v in metrics.items() if k != 'confusion_matrix'},
                    "Epoch": epoch
                })


                #сохранение модели с лучшей точностью
            if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

#сохранение чекпоинта после каждой эпохи
        save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir)
        torch.cuda.empty_cache()
        gc.collect()
    #время обучения
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    #построение графиков
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

#Запуск обучения с продолжением
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epoch, start_epoch=start_epoch)
#Сохранение лучшей модели
result_dir_s = os.path.join(result_dir, 'best_ensemble_model.ckpt')
torch.save(model_ft.state_dict(), result_dir_s)
wandb.finish()
