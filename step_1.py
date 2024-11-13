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


from fordata1 import *

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark = True
gc.collect()


torch.cuda.empty_cache()
##поменяла размер батча##
batch_sz = 16
num_epoch = 5
init_learning_rate = 0.0001
learning_rate_decay_factor = 0.2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EnsembleModel(nn.Module):
    def __init__(self, class_num):
        super(EnsembleModel, self).__init__()
        self.resnet = models.resnet50()
        num_ftrs_resnet = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.vit = models.vit_b_16()
        num_ftrs_vit = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Identity()
        self.classifier = nn.Linear(num_ftrs_resnet + num_ftrs_vit, class_num)

    def forward(self, x):
        resnet_out = self.resnet(x)
        vit_out = self.vit(x)
        combined = torch.cat((resnet_out, vit_out), dim=1)
        out = self.classifier(combined)
        return out

model_ft = EnsembleModel(class_num).to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=init_learning_rate)
# поменяла ReduceLROnPlateau
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=learning_rate_decay_factor, patience=10, verbose=True)

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

                torch.cuda.empty_cache()
                del inputs, labels, outputs, preds

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)

            # поправила метрики#####
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

            #сохранение лучшей 
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                scheduler.step(epoch_loss)
##добавила##
        torch.cuda.empty_cache()
        gc.collect()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

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

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epoch)
result_dir_s = os.path.join(result_dir, 'best_ensemble_model.ckpt')
torch.save(model_ft.state_dict(), result_dir_s)
