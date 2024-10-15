import weakref

import numpy as np


class PlainCosineScheduler(object):
    def __init__(
        self,
        klass,
        key,
        warmup_iters,
        total_iters,
        overwrite=False,
        init_value=None,
        base_value=None,
        final_value=None,
        step_init=-1,
    ):
        super().__init__()
        self.iter = step_init
        self.overwrite = overwrite
        self.base_value = base_value
        self.init_value = init_value if init_value is not None else base_value
        self.final_value = final_value
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.key = key
        self.klass = klass
        self.schedulers = [self.get_scheduler()]

    def get_scheduler(self):
        init_value = self.init_value
        base_value = self.base_value
        final_value = self.final_value
        warmup_iters = self.warmup_iters
        total_iters = self.total_iters

        # normalize in 0,1, then apply function (power) and denormalize
        normalized_schedule = np.linspace(0, 1, warmup_iters, endpoint=True)
        normalized_schedule = np.power(normalized_schedule, 1)
        warmup_schedule = (base_value - init_value) * normalized_schedule + init_value

        # main scheduling
        iters = np.arange(total_iters - warmup_iters + 1)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / (len(iters) - 1))
        )
        return np.concatenate((warmup_schedule, schedule))

    def step(self):
        self.iter = self.iter + 1
        vals = self[self.iter]
        for i, val in enumerate(vals):
            setattr(self.klass, self.key, val)

    def __getitem__(self, it):
        it = min(it, self.total_iters)
        return [scheduler[it] for scheduler in self.schedulers]


class CosineScheduler(object):
    def __init__(
        self,
        optimizer,
        warmup_iters,
        total_iters,
        key,
        overwrite=False,
        init_value=None,
        base_value=None,
        final_value=None,
        step_init=-1,
    ):
        super().__init__()
        self.iter = step_init
        self.overwrite = overwrite
        self.optimizer = optimizer
        self.base_value = base_value
        self.init_value = init_value
        self.final_value = final_value
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.key = key
        self.schedulers = [
            self.get_schedulers(group) for group in optimizer.param_groups
        ]

    def get_schedulers(self, group):
        init_value = group.get(self.key + "_init", self.init_value)
        base_value = group.get(self.key + "_base", self.base_value)
        final_value = group.get(self.key + "_final", self.final_value)
        warmup_iters = self.warmup_iters
        total_iters = self.total_iters
        if self.overwrite:
            final_value = self.final_value

        # normalize in 0,1, then apply function (power) and denormalize
        normalized_schedule = np.linspace(0, 1, warmup_iters, endpoint=True)
        normalized_schedule = np.power(normalized_schedule, 1)
        warmup_schedule = (base_value - init_value) * normalized_schedule + init_value

        # main scheduling
        iters = np.arange(total_iters - warmup_iters + 1)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / (len(iters) - 1))
        )
        return np.concatenate((warmup_schedule, schedule))

    def step(self):
        self.iter = self.iter + 1
        vals = self[self.iter]
        for group, val in zip(self.optimizer.param_groups, vals):
            if isinstance(group[self.key], (tuple, list)):
                val = (val, *group[self.key][1:])
            group[self.key] = val

    def __getitem__(self, it):
        it = min(it, self.total_iters)
        return [scheduler[it] for scheduler in self.schedulers]

    def get(self):
        return [group[self.key] for group in self.optimizer.param_groups]
