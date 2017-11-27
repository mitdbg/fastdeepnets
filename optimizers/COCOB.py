import math
import torch
from torch.optim import Optimizer


class COCOB(Optimizer):
    """Implements COCOB-Backprop algorithm.
    It has been proposed in `Backprop without Learning Rates Through Coin Betting`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
            alpha: (float, optional): constant to mitigate catastrophic forgetting at the beginning of the training (default: 100)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

        .. _Backprop without Learning Rates Through Coin Betting
            https://arxiv.org/abs/1705.07795
    """

    def __init__(self, params, alpha=100, weight_decay=0):
        defaults = dict(alpha=alpha, weight_decay=weight_decay)
        super(COCOB, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Current implementation does not support sparse gradient')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['L'] = p.data.new(p.size()).zero_().add_(1e-8)
                    state['grad_norm_sum'] = p.data.new(p.size()).zero_()
                    state['gradients_sum'] = p.data.new(p.size()).zero_()
                    state['tilde_w'] = p.data.new(p.size()).zero_()
                    state['reward'] = p.data.clone()

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                
                gradients_sum = state['gradients_sum']
                grad_norm_sum = state['grad_norm_sum']
                tilde_w = state['tilde_w']
                L = state['L']
                reward = state['reward']
                alpha = group['alpha']

                abs_grad = torch.abs(grad)
                L_update = torch.max(L, abs_grad)
                gradients_sum_update = gradients_sum + grad
                grad_norm_sum_update = grad_norm_sum + abs_grad
                reward_update = torch.clamp(reward - grad * tilde_w, min=0)
                new_w = -gradients_sum_update / (L_update * (torch.max(grad_norm_sum_update + L_update, alpha*L_update))) * (reward_update + L_update)
                var_update = p.data - tilde_w + new_w
                tilde_w_update = new_w

                gradients_sum.copy_(gradients_sum_update)
                grad_norm_sum.copy_(grad_norm_sum_update)
                p.data.copy_(var_update)
                tilde_w.copy_(tilde_w_update)
                L.copy_(L_update)
                reward.copy_(reward_update)

        return loss
