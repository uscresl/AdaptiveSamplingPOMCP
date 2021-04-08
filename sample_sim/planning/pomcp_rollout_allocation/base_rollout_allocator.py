import logging

import abc
from smallab.utilities.tqdm_to_logger import TqdmToLogger
from tqdm import tqdm


class BaseRolloutAllocator(abc.ABC):
    @abc.abstractmethod
    def should_continue(self) -> bool:
        pass

    def set_rewards(self, rewards):
        self.rewards = rewards


class AllocatorWhichChoosesRootActionMixin(abc.ABC):
    @abc.abstractmethod
    def get_action(self):
        pass


class PrecalculatedBaseRolloutAllocator(BaseRolloutAllocator):
    def __init__(self, logger=None, pbar=False):
        self.used_rollouts_this_iteration = 0
        self.currently_allocated_rollouts = self.allocated_rollouts()
        self.pbar = None
        self.should_create_pbar = False
        if pbar == True:
            assert logger is not None
            self.logger = logger
            self.should_create_pbar = True

    def should_continue(self):
        if self.should_create_pbar:
            self.pbar = tqdm("Rollouts", file=TqdmToLogger(logging.getLogger(self.logger)),
                             total=self.currently_allocated_rollouts)
            self.should_create_pbar = False
        if self.pbar is not None:
            self.pbar.update(1)
        self.used_rollouts_this_iteration += 1
        return self.used_rollouts_this_iteration < self.currently_allocated_rollouts

    @abc.abstractmethod
    def allocated_rollouts(self):
        pass
