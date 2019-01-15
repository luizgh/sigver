from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


def squared_l2_norm(x: torch.Tensor) -> torch.Tensor:
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return squared_l2_norm(x).sqrt()


def restrict_l2_norm(delta: torch.Tensor, max_l2: float) -> torch.Tensor:
    # Project delta to the feasible region
    delta_normalized = delta * max_l2 / l2_norm(delta).view(-1, 1, 1, 1)
    sign = torch.sign(delta)
    delta = torch.min(torch.abs(delta), torch.abs(delta_normalized)) * sign
    return delta


class PGD_max_l2_truncate:
    """
    PGD attack with box constraints on the delta (image_constraints),
    plus truncating delta to a maximum L2 norm in each step
    Args:
        steps (int): number of search steps
        max_l2 (float): maximum l2 norm for the noise
        num_classes (int, optional): number of classes of the model to attack. Default: 1000
        image_constraints (tuple, optional): bounds of the images. Default: (0, 1)
        fast (bool, optional): whether to use Adam for the optimization - usually faster but L2 is bigger. Default: False
        device (torch.device, optional): device to use for the attack. Default: torch.device('cpu')
        callback (object, optional): callback to display losses
    """

    # TODO: update docstring
    def __init__(self,
                 steps: int,
                 max_l2: float,
                 image_constraints: Tuple[float, float] = (0, 1),
                 num_classes: int = 1000,
                 fast: bool = False,
                 device: torch.device = torch.device('cpu'),
                 callback: Optional = None) -> None:
        self.steps = steps
        self.num_classes = num_classes
        self.fast = fast
        self.callback = callback

        self.max_l2 = max_l2
        self.max_l2_squared = max_l2 ** 2

        self.boxmin = image_constraints[0]
        self.boxmax = image_constraints[1]

        self.device = device

    def attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
               targeted: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs.
        Args:
            model (nn.Module): model to attack
            inputs (torch.Tensor): batch of images to generate adv for
            labels (torch.Tensor): true labels in case of untargeted, target in case of targeted
            targeted (bool): whether to perform a targeted attack or not
        """
        multiplier = 1 if targeted else -1
        delta = torch.zeros_like(inputs, requires_grad=True)

        # Setup optimizer
        if self.fast:
            optimizer = optim.SGD([delta], lr=1)
            final_lr = 0.01
            linear = [lambda iter: (self.steps - iter) / self.steps * (1 - final_lr) + final_lr]
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, linear)
        else:
            optimizer = optim.SGD([delta], lr=1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.steps // 4, gamma=0.3)

        for i in range(self.steps):
            scheduler.step()

            adv = inputs + delta

            logits = model(adv)
            ce_loss = F.cross_entropy(logits, labels)

            loss = multiplier * ce_loss

            optimizer.zero_grad()
            loss.backward()
            if self.fast:
                delta.grad.div_(delta.grad.view(len(delta), -1).norm(p=2, dim=1).view(-1, 1, 1, 1))
            optimizer.step()

            if self.callback:
                self.callback.scalar('ce_loss', i, ce_loss)

            new_delta = torch.clamp(inputs + delta, self.boxmin, self.boxmax) - inputs
            new_delta = restrict_l2_norm(new_delta, self.max_l2)
            delta.data.copy_(new_delta)

        return (inputs + delta).detach()
