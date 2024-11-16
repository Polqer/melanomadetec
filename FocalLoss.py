import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma  # Параметр фокуса
        self.alpha = alpha  # Коэффициент веса
        self.reduction = reduction  # Способ агрегации потерь (среднее или сумма)

    def forward(self, inputs, targets):
        # Преобразуем вероятности в логарифмы
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Получаем вероятности для правильных классов
        probs = torch.exp(log_probs)
        target_log_probs = log_probs.gather(1, targets.view(-1, 1))

        # Вычисляем Focal Loss
        loss = -self.alpha * (1 - probs) ** self.gamma * target_log_probs
        
        # Суммируем или усредняем по всей выборке в зависимости от reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
criterion = nn.CrossEntropyLoss()