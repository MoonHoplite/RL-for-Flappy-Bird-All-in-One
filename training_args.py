import torch
from typing import Optional


class TrainingArguments:
    def __init__(self, **kwargs):
        
        self.input_shape: tuple = (4, 80, 80)   # The shape of "states"
        self.actions: int = 2   # The number of actions
        self.initial_epsilon: float = 1 # Initial value of ε-greedy strategy, will decay linearly to final epsilon
        self.final_epsilon: float = 0.00001 # Final value of ε-greedy strategy
        self.explore: int = 100000  # Explore steps (frames) for DQN
        self.observe: int = 20000   # Observe steps (frames) for DQN
        self.replay_memory: int = 50000 # Max number of replay steps for DQN
        self.synchronize_steps: int = 1000  # Steps of
        self.jump_prob: float = 0.1 # Jump probability for DQN during observation stage
        self.batch_size: int = 512  # Also is the maximum number of replay for VPG and PPO
        self.gamma: float = 0.99    # Discount factor for DQN, VPG and PPO
        self.save_interval: int = 20000 # Auto save interval. The unit is step (frame) for DQN, epoch for VPG and PPO
        self.load_checkpoint: Optional[str] = None  # Specify a path for saved model. Should include both model parameters and optimizer states 
        self.max_save: int = 5  # Max number of saves. The earliest save will be deleted if exceeded.
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"   # Default device
        self.dqn_dueling: bool = True  # Whether to use Dueling DQN
        self.lr_value: float = 1e-5  # Learning rate for value model in DQN, VPG and PPO
        self.lr_policy: float = 1e-6   # Learning rate for policy model in VPG and PPO
        self.max_t: int = 1000000    # Max number of steps (frames) for DQN
        self.lam: float = 0.95 # Discount factor used in GAE
        self.num_epochs: int = 10000 # Number of epochs for VPG and PPO
        self.clip_ratio: float = 0.1   # Clip ratio for PPO
        
        self.num_policy_updates: int = 2 # Number of policy model updates for every batch of data
        self.num_value_updates: int = 2  # Number of value model updates for every batch of data

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'TrainingArguments' object has no attribute '{key}'")

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    def __repr__(self):
        return str(self.to_dict())