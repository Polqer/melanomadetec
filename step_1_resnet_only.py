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
from fordata1 import *
###  wandb.init(project="my-awesome-project", settings=wandb.Settings(init_timeout=300))###


#Модель на основе ResNet152
class ResNetModel(nn.Module):
    def __init__(self, class_num):
        super(ResNetModel, self).__init__()
        #ResNet152
        self.resnet = models.resnet152(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, class_num)  # Заменяем последний слой

    def forward(self, x):
        # Прямой проход через ResNet152
        out = self.resnet(x)
        return out
    

#только реснет
model_ft = ResNetModel(class_num).to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=init_learning_rate)

exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=learning_rate_decay_factor, patience=10)

def train_model(model, criterion, optimizer, scheduler, num_epochs=num_epoch):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            
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
                running_corrects += torch.sum(preds == labels.data)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

      
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # # Расчет чувствительности и специфичности для каждого класса
            # if phase == 'val':
            #     cm = confusion_matrix(all_labels, all_preds)
            #     sensitivity = cm.diagonal() / cm.sum(axis=1)
            #     specificity = (cm.sum() - cm.sum(axis=0) - cm.sum(axis=1) + cm.diagonal()) / (cm.sum() - cm.sum(axis=1))
            #     print('Sensitivity:', sensitivity)
            #     print('Specificity:', specificity)
        
        
            if phase == 'val':
              cm = confusion_matrix(all_labels, all_preds)
              sensitivity = []
              specificity = []
              num_classes = cm.shape[0]  

              for i in range(num_classes):
                  TP = cm[i, i]
                  FN = cm[i, :].sum() - TP
                  FP = cm[:, i].sum() - TP
                  TN = cm.sum() - (TP + FN + FP)

    #Чувствительность 
                  sensitivity_class = TP / (TP + FN) if (TP + FN) > 0 else 0
                  sensitivity.append(sensitivity_class)

    #Специфичность 
                  specificity_class = TN / (TN + FP) if (TN + FP) > 0 else 0
                  specificity.append(specificity_class)
              print("\nMetrics per class:")
              for i, class_name in enumerate(class_names):
                  print(f"Class: {class_name}")
                  print(f"  Sensitivity: {sensitivity[i]:.4f}")
                  print(f"  Specificity: {specificity[i]:.4f}")

#Расчет чувствительности  и F1-меры для каждого класса
            if phase == 'val':
              recall = recall_score(all_labels, all_preds, average='macro', zero_division=1)
              f1 = f1_score(all_labels, all_preds, average='macro', zero_division=1)
              print(f'Recall: {recall:.4f}')
              print(f'F1-score: {f1:.4f}')
                #Сохранение модели с лучшей точностью
            if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
#Сохранение чекпоинта
        checkpoint_path = os.path.join(result_dir, f'checkpoint_epoch_{epoch+1}.ckpt')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
        torch.cuda.empty_cache()
        gc.collect()
        
        # тот самый wandb, который не хочет работать
    #         if phase == 'train':
    #             wandb.log({
    #     "Train Loss": epoch_loss,
    #     "Train Accuracy": epoch_acc,
    #     "Train Precision": epoch_precision,
    #     "Train Recall": epoch_recall,
    #     "Train F1": epoch_f1,
    #     "Train AUC": epoch_auc,
    #     "Epoch": epoch
    # })
    #         else:
    #             wandb.log({
    #     "Val Loss": epoch_loss,
    #     "Val Accuracy": epoch_acc,
    #     "Val Precision": epoch_precision,
    #     "Val Recall": epoch_recall,
    #     "Val F1": epoch_f1,
    #     "Val AUC": epoch_auc,
    #     "Epoch": epoch
    # })
    
    
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

#запуск обучения
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epoch)

#Сохранение лучшей модели
result_dir_s = os.path.join(result_dir, 'best_resnet_model.ckpt')
torch.save(model_ft.state_dict(), result_dir_s)
