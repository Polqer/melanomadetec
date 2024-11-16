import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma  
        self.alpha = alpha  
        self.reduction = reduction  
    def forward(self, inputs, targets):
    
        log_probs = F.log_softmax(inputs, dim=1)
        
        probs = torch.exp(log_probs)
        target_log_probs = log_probs.gather(1, targets.view(-1, 1))

        loss = -self.alpha * (1 - probs) ** self.gamma * target_log_probs
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
criterion = nn.CrossEntropyLoss()