import torch.optim as optim
from typing import Optional

class Superconvergence(optim.lr_scheduler._LRScheduler):

    def __init__(self,
                optimizer,
                max_lr,
                epoch_annihilation: int,
                start_lr_fraction: int=50,
                last_epoch: int=-1):
        """
            PyTorch implementation of Triangle Scheduler with annihilation phase as in
            Superconvergence paper https://arxiv.org/abs/1708.07120
            Developer's note:
            if cyclic momentum would be implemented, according to paper
            0.85 as min val works just fine -> Take that value!
        """
        self.max_lr = max_lr
        self.init_lr = max_lr / start_lr_fraction
        x_steps = epoch_annihilation / 2
        self.epoch_annihilation = epoch_annihilation
        self.lr_step = (self.max_lr - self.init_lr) / x_steps
        
        super(Superconvergence, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch >= self.epoch_annihilation:
            new_lr = self.init_lr - ((self.last_epoch - self.epoch_annihilation) * 0.01 * self.lr_step)
            new_lr = (new_lr if new_lr > 1e-8 else 1e-8)
        elif self.last_epoch < (self.epoch_annihilation / 2):
            new_lr =  self.init_lr + (self.last_epoch * self.lr_step)
        else:
            new_lr =  self.max_lr - (self.last_epoch - ((self.epoch_annihilation / 2) * self.lr_step))
        return [new_lr for group in self.optimizer.param_groups]

class Trapezoid(optim.lr_scheduler._LRScheduler):

    def __init__(self,
                optimizer,
                n_iterations: int,
                max_lr: float,
                start_lr: Optional[float]=None,
                annihilate: bool=True,
                last_epoch: int=-1
                ):
        """
            PyTorch implementation of Trapezoid Scheduler https://arxiv.org/abs/1802.08770
            Lazy: n_iterations is the total amount of iterations that this scheduler will be used for!
        """
        

        self.n_iters = n_iterations
        self.max_lr = max_lr
        if start_lr is None:
            self.start_lr = max_lr / 10
        else:
            self.start_lr = start_lr
        self.stop_warmup = int(0.1 * n_iterations)
        self.start_decline = int(0.8 * n_iterations)
        self.start_annihilate = int(0.95 * n_iterations) if annihilate else n_iterations

        super(Trapezoid, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch < self.stop_warmup:
            step_size = (self.max_lr - self.start_lr) / self.stop_warmup
            new_lr = self.start_lr + step_size * self.last_epoch
        elif self.last_epoch < self.start_decline:
            new_lr = self.max_lr
        elif self.last_epoch <= self.start_annihilate:
            step_size = (self.max_lr - self.start_lr) / (self.start_annihilate - self.start_decline)
            new_lr = self.max_lr - step_size * (self.last_epoch - self.start_decline)
        else:
            step_size = (self.start_lr - self.start_lr / 20) / (self.n_iters - self.start_annihilate)
            new_lr = self.start_lr - step_size * (self.last_epoch - self.start_annihilate)
            
        return [new_lr for group in self.optimizer.param_groups]
