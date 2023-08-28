import torch.optim as optim
import math


class WarmUpScheduler:

    def __init__(self, optimizer, params, max_lr):
        """Implements learning rate warm up for transformer post norm

        Args:
            optimizer:
            params: [warmup_steps, num_decay_steps, decay_factor]
            max_lr:
        """
        self.optimizer = optimizer

        self.warmup_steps = params[0]
        if len(params) == 1:
            self.gamma = 1.0
        else:
            self.gamma = math.exp(math.log(params[2]) / params[1])
        self.max_lr = max_lr
        self._step = 0
        self._lr = 0  # Current last learning rate

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self):
        "Update parameters and rate"
        self._step += 1
        lr = self.compute_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self._lr = lr
        self.optimizer.step()

    def compute_lr(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step

        if step < self.warmup_steps:  # Warmup phase
            return min(step / self.warmup_steps, 1.0) * self.max_lr
        else:  # decay phase
            return math.pow(self.gamma, step - self.warmup_steps) * self.max_lr

    def get_last_lr(self):
        return [self._lr]

    def __repr__(self):
        return f'WarmUpScheduler with (warmup_steps={self.warmup_steps}, max lr={self.max_lr})'


def fetch_optimizer(cfg, model):
    # to be compatible with warm_up
    if cfg.scheduler_name == "warm_up":
        learning_rate = 0.0  # start from 0
    else:
        learning_rate = cfg.learning_rate

    if cfg.optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif cfg.optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError("Unknown optimizer type: {}.".format(cfg.optimizer_name))

    if cfg.scheduler_name == "exp":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)
    elif cfg.scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.decay_steps, gamma=cfg.gamma)
    elif cfg.scheduler_name == "multi_step":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma)
    elif cfg.scheduler_name == "warm_up":
        # warm up, then smooth exponential decay
        scheduler = WarmUpScheduler(optimizer, params=[cfg.warmup_steps, cfg.decay_steps, cfg.gamma], max_lr=cfg.learning_rate)
    elif cfg.scheduler_name == "none" or cfg.scheduler_name is None:
        # no decay
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=1.0)
    else:
        raise NotImplementedError("Unknown scheduler type: {}.".format(cfg.scheduler_name))
    return optimizer, scheduler
