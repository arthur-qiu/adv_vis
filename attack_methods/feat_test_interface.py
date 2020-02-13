import torch
import torch.nn as nn
import torch.nn.functional as F
from attack_methods import pgd, feature_targets


class Restart_PGD(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, num_restart = 10, data_min = -1.0, data_max = 1.0):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.num_restart = num_restart
        self.data_min = data_min
        self.data_max = data_max
        self.untargeted_pgd = pgd.PGD(epsilon=epsilon, num_steps=num_steps, step_size=step_size, data_min=data_min, data_max=data_max).cuda()

    def forward(self, model, bx, by):
        final_results = torch.zeros_like(by).byte() + 1
        for re in range(self.num_restart):
            adv_bx = self.untargeted_pgd(model, bx, by)
            logits = model(adv_bx)
            if len(logits) == 2:
                logits = logits[1]
            pred = logits.data.max(1)[1]
            correct = pred.eq(by.data)
            final_results &= correct
            if re == 0:
                single_correct = final_results.sum().item()
        final_correct = final_results.sum().item()

        return final_correct, single_correct

class Mult_Targets(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, num_classes = 10, data_min = -1.0, data_max = 1.0):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.num_classes = num_classes
        self.data_min = data_min
        self.data_max = data_max
        self.targeted_pgd = pgd.PGD_Margin_Target(epsilon=epsilon, num_steps=num_steps, step_size=step_size, data_min=data_min, data_max=data_max).cuda()

    def forward(self, model, bx, by):
        final_results = torch.zeros_like(by).byte() + 1
        for re in range(1, self.num_classes):
            adv_bx = self.targeted_pgd(model, bx, by, (by+re)%self.num_classes)
            logits = model(adv_bx)
            if len(logits) == 2:
                logits = logits[1]
            pred = logits.data.max(1)[1]
            correct = pred.eq(by.data)
            final_results &= correct
            if re == 1:
                single_correct = final_results.sum().item()
        final_correct = final_results.sum().item()

        return final_correct, single_correct

class Feature_Attack(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, num_classes = 10, data_min = -1.0, data_max = 1.0):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.num_classes = num_classes
        self.data_min = data_min
        self.data_max = data_max
        self.feature_attack = feature_targets.Feature_Targets(epsilon=epsilon, num_steps=num_steps, step_size=step_size, data_min=data_min, data_max=data_max).cuda()

    def forward(self, model, bx, by, target_bx):
        final_results = torch.zeros_like(by).byte() + 1
        for re in range(target_bx.shape[0]):
            adv_bx = self.feature_attack(model, bx, by, torch.cat((target_bx[re:], target_bx[:re]),0))
            logits = model(adv_bx)
            if len(logits) == 2:
                logits = logits[1]
            pred = logits.data.max(1)[1]
            correct = pred.eq(by.data)
            final_results &= correct
            if re == 0:
                single_correct = final_results.sum().item()
        final_correct = final_results.sum().item()

        return final_correct, single_correct

class Test_All(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, data_min = -1.0, data_max = 1.0):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.data_min = data_min
        self.data_max = data_max
        self.untargeted_pgd = pgd.PGD(epsilon=epsilon, num_steps=num_steps, step_size=step_size, data_min=data_min, data_max=data_max).cuda()
        self.targeted_pgd = pgd.PGD(epsilon=epsilon, num_steps=num_steps, step_size=step_size, data_min=data_min, data_max=data_max).cuda()
        self.feature_attack = feature_targets.Feature_Targets(epsilon=epsilon, num_steps=num_steps, step_size=step_size, data_min=data_min, data_max=data_max).cuda()


    def forward(self, model, bx, by, target_bx):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """


        adv_bx = bx.detach().clone()
        # TODO
        # if self.random_start:
        #     adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon, self.epsilon)
        #     adv_bx = adv_bx.clamp(self.data_min, self.data_max)
        #
        # for i in range(self.num_steps):
        #     adv_bx.requires_grad_()
        #     with torch.enable_grad():
        #         logits = model(adv_bx)
        #         if len(logits) == 2:
        #             logits = logits[1]
        #         loss = F.cross_entropy(logits, by, reduction='sum')
        #     grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]
        #
        #     if self.grad_sign:
        #         adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
        #
        #     adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(self.data_min, self.data_max)

        return adv_bx


