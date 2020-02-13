import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_l2(x):
    """
    Expects x.shape == [N, C, H, W]
    """
    norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1)
    norm = norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return x / norm

def pair_cos_dist(x, y):
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    c = torch.clamp(1 - cos(x, y), min=0)
    return c

class Feature_Targets(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, data_min = -1.0, data_max = 1.0, grad_sign=True, random_start = True):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign
        self.data_min = data_min
        self.data_max = data_max
        self.random_start = random_start

    def forward(self, model, bx, by, target_bx):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        adv_bx = bx.detach().clone()
        target = target_bx.detach().clone()
        if self.random_start:
            adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon, self.epsilon)
            adv_bx = adv_bx.clamp(self.data_min, self.data_max)

        target_feature, target_logits = model(target)

        for i in range(self.num_steps):
            adv_bx.requires_grad_()
            with torch.enable_grad():
                feature, logits = model(adv_bx)
                loss = pair_cos_dist(feature, target_feature).mean()
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

            if self.grad_sign:
                adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
            else:
                grad = normalize_l2(grad.detach())
                adv_bx = adv_bx.detach() + self.step_size * grad

            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(self.data_min, self.data_max)

        return adv_bx