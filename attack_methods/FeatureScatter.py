import torch
import torch.nn as nn

from attack_methods import ot

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def normalize_l2(x):
    """
    Expects x.shape == [N, C, H, W]
    """
    norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1)
    norm = norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return x / norm

class FeatureScatter(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, data_min = -1.0, data_max = 1.0, grad_sign=True, random_start = True):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign
        self.data_min = data_min
        self.data_max = data_max
        self.random_start = random_start

    def forward(self, model, bx, by):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        adv_bx = bx.detach().clone()
        batch_size = adv_bx.size(0)
        m = batch_size
        n = batch_size

        logits_nature = model(adv_bx)
        if len(logits_nature) == 2:
            logits_nature = logits_nature[1]

        if self.random_start:
            adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon, self.epsilon)
            adv_bx = adv_bx.clamp(self.data_min, self.data_max)

        for i in range(self.num_steps):
            adv_bx.requires_grad_()
            with torch.enable_grad():
                logits = model(adv_bx)
                if len(logits) == 2:
                    logits = logits[1]

                loss = ot.sinkhorn_loss_joint_IPOT(1, 0.00, logits_nature,
                                                  logits, None, None,
                                                  0.01, m, n)
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

            if self.grad_sign:
                adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
            else:
                grad = normalize_l2(grad.detach())
                adv_bx = adv_bx.detach() + self.step_size * grad

            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(self.data_min, self.data_max)

        return adv_bx
