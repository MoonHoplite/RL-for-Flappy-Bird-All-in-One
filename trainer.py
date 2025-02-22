import torch
from torch import nn, Tensor
from typing import Tuple
import numpy as np
from abc import ABC, abstractmethod
from game import wrapped_flappy_bird as game
import model
from copy import deepcopy
import cv2
from collections import deque
import random
import os
import scipy
from training_args import TrainingArguments


class RLAlgorithms(ABC):
    def __init__(self, config: TrainingArguments) -> None:
        self.config = config
    
    
    @abstractmethod
    def train(self) -> None:
        pass
    
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame_shape = self.config.input_shape[1:]
        frame = cv2.cvtColor(cv2.resize(frame, frame_shape), cv2.COLOR_BGR2GRAY)
        _, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        return frame
        
    
class DQN(RLAlgorithms):
    def __init__(self, config):
        super().__init__(config)
        
        # Load config
        self.config = config
        
        # Initialize policy net, optimizer and criterion
        self.policy_net = model.DQN_q(input_shape=self.config.input_shape, output_shape = self.config.actions, dueling=self.config.dqn_dueling)
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.config.lr_policy)
        self.criterion = nn.MSELoss()
        
        # Load checkpoint
        if self.config.load_checkpoint is not None:
            try:
                chk = torch.load(self.config.load_checkpoint, weights_only=True)
                self.policy_net.load_state_dict(chk['model_state_dict'])
                self.optimizer.load_state_dict(chk['optimizer_state_dict'])
                print("checkpoint loaded successfully")
            except:
                print("checkpoint failed to load")
    
        # Initialize target net for soft update
        self.target_net = deepcopy(self.policy_net)
        self.target_net.eval()
        
        # Move to device
        self.policy_net.to(self.config.device)
        self.target_net.to(self.config.device)
        
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.config.device)
        
        # Initialize game instance
        self.game_state = game.GameState()
        
        # Initialize initial epsilon
        self.epsilon = self.config.initial_epsilon
        
        
    def select_action(self, state: torch.Tensor) -> int:
        # Select random action
        if np.random.rand() < self.epsilon:
            action = 1 if np.random.rand() < self.config.jump_prob else 0
            
        # Select action that maximize Q value
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.config.device)
                q_value = self.policy_net(state)
                action = torch.argmax(q_value).item()
        return action
    
    
    def train(self):
        # Generate first frame
        a_t = 0
        x_t, _, _ = self.game_state.frame_step(a_t)
        x_t = self.process_frame(x_t)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2).transpose((2, 0, 1))
        
        # Set up a replay buffer
        replay_buffer = deque(maxlen=self.config.replay_memory)
        save_list = deque()
        
        # Initialize
        t = 0
        episode = 0
        total_reward = 0
        
        for t in range(self.config.max_t):
            a_t = self.select_action(s_t)
            if self.epsilon > self.config.final_epsilon and t > self.config.observe:
                self.epsilon -= (self.config.initial_epsilon - self.config.final_epsilon) / (self.config.explore)

            x_t1, r_t, terminal = self.game_state.frame_step(a_t)
            total_reward += r_t
            x_t1 = self.process_frame(x_t1)
            s_t1 = np.append(s_t[1:], x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1]), axis=0)
            
            replay_buffer.append((s_t, a_t, r_t, s_t1, terminal))
            
            if t > self.config.observe and len(replay_buffer) >= self.config.batch_size:
                minibatch = random.sample(replay_buffer, self.config.batch_size)
                
                state_batch = torch.tensor(np.array([d[0] for d in minibatch]), dtype=torch.float32).to(self.config.device)
                action_batch = torch.tensor(np.array([d[1] for d in minibatch]), dtype=torch.int64).to(self.config.device)
                reward_batch = torch.tensor(np.array([d[2] for d in minibatch]), dtype=torch.float32).to(self.config.device)
                next_state_batch = torch.tensor(np.array([d[3] for d in minibatch]), dtype=torch.float32).to(self.config.device)
                terminal_batch = torch.tensor(np.array([d[4] for d in minibatch]), dtype=torch.float32).to(self.config.device)
                
                cur_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
                next_q_values = self.target_net(next_state_batch).max(1)[0]
                expected_q_values = reward_batch + self.config.gamma * next_q_values * (1 - terminal_batch)
                
                loss = self.criterion(cur_q_values, expected_q_values)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            s_t = s_t1
            t += 1
            
            # Update target network
            if t % self.config.synchronize_steps == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.target_net.eval()
            
            # Print info every episode
            if terminal:
                state = "observe" if t <= self.config.observe else "explore" if t <= self.config.observe + self.config.explore else "train"
                print(f"TIMESTEP {t} / EPISODE {episode} / STATE {state} / EPSILON {self.epsilon:.4f} / REWARD {total_reward:.1f}")
                total_reward = 0
                episode += 1
            
            # Save model regularly
            if t % self.config.save_interval == 0:
                torch.save(
                    {
                    'optimizer_state_dict': self.optimizer.state_dict(), 'model_state_dict': self.policy_net.state_dict()
                    }, 
                    f"saved_models/flappybird_dqn_{t}.pth"
                    )
                save_list.append(f"saved_models/flappybird_dqn_{t}.pth")
                if len(save_list) > self.config.max_save:
                    os.remove(save_list.popleft())
    
    
    def process_frame(self, frame):
        return super().process_frame(frame)
    
    
class VPG(RLAlgorithms):
    def __init__(self, config: TrainingArguments) -> None:
        super().__init__(config)
        
        # Load config
        self.config = config
        
        # Initialize policy net, optimizer and criterion
        self.value_net = model.DQN_q(input_shape=self.config.input_shape, output_shape = 1)
        self.value_optimizer = torch.optim.AdamW(self.value_net.parameters(), lr=self.config.lr_value)
        
        self.policy_net = model.VPG_p(input_shape=self.config.input_shape, output_shape = self.config.actions)
        self.policy_optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr = self.config.lr_policy)
        
        
        # Load checkpoint
        if self.config.load_checkpoint is not None:
            # load_checkpoint must be a tuple of (actor model, critic model)
            assert len(self.config.load_checkpoint) == 2
            try:
                actor_chk = torch.load(self.config.load_checkpoint[0], weights_only=True)
                self.policy_net.load_state_dict(actor_chk['model_state_dict'])
                self.policy_optimizer.load_state_dict(actor_chk['optimizer_state_dict'])
                critic_chk = torch.load(self.config.load_checkpoint[1], weights_only=True)
                self.value_net.load_state_dict(critic_chk['model_state_dict'])
                self.value_optimizer.load_state_dict(critic_chk['optimizer_state_dict'])
                print("checkpoint loaded successfully")
            except:
                print("checkpoint failed to load")
            
        # Move to device
        for state in self.value_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.config.device)
        for state in self.policy_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.config.device)
    
        self.policy_net.to(self.config.device)
        self.value_net.to(self.config.device)
        
        # Initialize game instance
        self.game_state = game.GameState()
        
    
    def policy_criterion(self, obs: Tensor, actions: Tensor, advantages: Tensor, log_probs_old: Tensor) -> Tuple[Tensor, float]:
        logits = self.policy_net(obs)
        log_probs = torch.log(logits.gather(1, actions.unsqueeze(1))).squeeze(1)
        kl = (log_probs_old - log_probs).mean().item()
        return -(log_probs * advantages).mean(), kl
    
    
    def value_criterion(self, obs: Tensor, returns: Tensor) -> Tensor:
        return ((self.value_net(obs) - returns)**2).mean()
    
    
    def train(self) -> None:

        save_list = deque()
        
        # Train for num_epochs
        for epoch in range(self.config.num_epochs):
            
            obs, actions, returns, advantages, log_probs, total_reward = self.sample_trajectory()
            obs = torch.tensor(obs, dtype=torch.float32).to(self.config.device)
            actions = torch.tensor(actions, dtype=torch.int64).to(self.config.device)
            returns = torch.tensor(returns, dtype=torch.float32).to(self.config.device)
            advantages = torch.tensor(advantages, dtype=torch.float32).to(self.config.device)
            log_probs = torch.tensor(log_probs, dtype=torch.float32).to(self.config.device)
            
            for _ in range(self.config.num_policy_updates):
                self.policy_optimizer.zero_grad()
                policy_loss, kl = self.policy_criterion(obs, actions, advantages, log_probs)
                policy_loss.backward()
                self.policy_optimizer.step()
            
            for _ in range(self.config.num_value_updates):
                self.value_optimizer.zero_grad()
                value_loss = self.value_criterion(obs, returns)
                value_loss.backward()
                self.value_optimizer.step()
            
            if epoch % self.config.save_interval == 0:
                torch.save(
                    {
                    'optimizer_state_dict': self.policy_optimizer.state_dict(), 'model_state_dict': self.policy_net.state_dict()
                    }, 
                    f"saved_models/flappybird_vpg_actor_{epoch}.pth"
                    )
                torch.save(
                    {
                    'optimizer_state_dict': self.value_optimizer.state_dict(), 'model_state_dict': self.value_net.state_dict()
                    }, 
                    f"saved_models/flappybird_vpg_critic_{epoch}.pth"
                    )
                save_list.append((f"saved_models/flappybird_vpg_actor_{epoch}.pth", f"saved_models/flappybird_vpg_critic_{epoch}.pth"))
                if len(save_list) > self.config.max_save:
                    a, b = save_list.popleft()
                    os.remove(a)
                    os.remove(b)
                    del a, b
            
            print(f"Epoch: {epoch}, Reward: {total_reward:.1f}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, KL Div: {kl:.4f}")

    
    def process_frame(self, frame):
        return super().process_frame(frame)
    
    
    def sample_trajectory(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        # Generate first frame
        action = 0
        obs, _, _ = self.game_state.frame_step(action)
        obs = self.process_frame(obs)
        obs = np.stack((obs, obs, obs, obs), axis=2).transpose(2, 0, 1)
        
        replay_buffer = Buffer(self.config.batch_size, self.config.input_shape, self.config.actions, self.config.gamma, self.config.lam)
        
        while True:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.config.device)
            action_dist = self.policy_net(obs_tensor)
            value = self.value_net(obs_tensor).item()
            
            action_dist = torch.distributions.Categorical(action_dist)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            action = action.item()
            obs_next_single, reward, terminal = self.game_state.frame_step(action)
            # for _ in range(1):
            #     _, _, _ = self.game_state.frame_step(0)
            obs_next_single = self.process_frame(obs_next_single)
            obs_next = np.append(obs[1:], obs_next_single.reshape(1, obs_next_single.shape[0], obs_next_single.shape[1]), axis=0)
            
            if replay_buffer.cur_idx < replay_buffer.max_size:
                replay_buffer.add(obs, action, reward, value, log_prob)
            else:
                return replay_buffer.get(replay_buffer.start_idx)
            obs = obs_next
            
            if terminal:
                replay_buffer.end_trajectory()
    
    
class PPO(RLAlgorithms):
    def __init__(self, config: TrainingArguments) -> None:
        super(PPO, self).__init__(config)
        # Load config
        self.config = config
        
        # Initialize policy net, optimizer and criterion
        self.value_net = model.DQN_q(input_shape=self.config.input_shape, output_shape = 1)
        self.value_optimizer = torch.optim.AdamW(self.value_net.parameters(), lr=self.config.lr_value)
        
        self.policy_net = model.VPG_p(input_shape=self.config.input_shape, output_shape = self.config.actions)
        self.policy_optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.config.lr_policy)
        
        
        # Load checkpoint
        if self.config.load_checkpoint is not None:
            # load_checkpoint must be a tuple of (actor model, critic model)
            assert len(self.config.load_checkpoint) == 2
            try:
                actor_chk = torch.load(self.config.load_checkpoint[0], weights_only=True)
                self.policy_net.load_state_dict(actor_chk['model_state_dict'])
                self.policy_optimizer.load_state_dict(actor_chk['optimizer_state_dict'])
                critic_chk = torch.load(self.config.load_checkpoint[1], weights_only=True)
                self.value_net.load_state_dict(critic_chk['model_state_dict'])
                self.value_optimizer.load_state_dict(critic_chk['optimizer_state_dict'])
                print("checkpoint loaded successfully")
            except:
                print("checkpoint failed to load")
            
        # Move to device
        for state in self.value_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.config.device)
        for state in self.policy_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.config.device)
    
        self.policy_net.to(self.config.device)
        self.value_net.to(self.config.device)
        
        # Initialize game instance
        self.game_state = game.GameState()
        
    
    def policy_criterion(self, obs: Tensor, actions: Tensor, advantages: Tensor, log_probs_old: Tensor, entropy_weight: float=0.01) -> Tuple[Tensor, float]:
        logits = self.policy_net(obs)
        action_dist = torch.distributions.Categorical(probs=logits)
        # log_probs = torch.log(logits.gather(1, actions.unsqueeze(1))).squeeze(1)
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean().item()
        ratio = torch.exp(log_probs - log_probs_old)
        clip_advantages = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
        kl = ((ratio - 1) - advantages).mean().item()
        return -(torch.min(ratio * advantages, clip_advantages)).mean(), kl, entropy
    
    
    def value_criterion(self, obs: Tensor, returns: Tensor) -> Tensor:
        return ((self.value_net(obs) - returns)**2).mean()
    
    
    def train(self) -> None:

        save_list = deque()
        
        # Train for num_epochs
        for epoch in range(self.config.num_epochs):
            
            obs, actions, returns, advantages, log_probs, total_reward = self.sample_trajectory()
            obs = torch.tensor(obs, dtype=torch.float32).to(self.config.device)
            actions = torch.tensor(actions, dtype=torch.int64).to(self.config.device)
            returns = torch.tensor(returns, dtype=torch.float32).to(self.config.device)
            advantages = torch.tensor(advantages, dtype=torch.float32).to(self.config.device)
            log_probs = torch.tensor(log_probs, dtype=torch.float32).to(self.config.device)
            
            for _ in range(self.config.num_policy_updates):
                self.policy_optimizer.zero_grad()
                policy_loss, kl, entropy = self.policy_criterion(obs, actions, advantages, log_probs)
                policy_loss.backward()
                self.policy_optimizer.step()
            
            for _ in range(self.config.num_value_updates):
                self.value_optimizer.zero_grad()
                value_loss = self.value_criterion(obs, returns)
                value_loss.backward()
                self.value_optimizer.step()
            
            if epoch % self.config.save_interval == 0:
                torch.save(
                    {
                    'optimizer_state_dict': self.policy_optimizer.state_dict(), 'model_state_dict': self.policy_net.state_dict()
                    }, 
                    f"saved_models/flappybird_ppo_actor_{epoch}.pth"
                    )
                torch.save(
                    {
                    'optimizer_state_dict': self.value_optimizer.state_dict(), 'model_state_dict': self.value_net.state_dict()
                    }, 
                    f"saved_models/flappybird_ppo_critic_{epoch}.pth"
                    )
                save_list.append((f"saved_models/flappybird_ppo_actor_{epoch}.pth", f"saved_models/flappybird_ppo_critic_{epoch}.pth"))
                if len(save_list) > self.config.max_save:
                    a, b = save_list.popleft()
                    os.remove(a)
                    os.remove(b)
                    del a, b
            
            print(f"Epoch: {epoch}, Reward: {total_reward:.1f}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, KL Div: {kl:.4f}, Entropy: {entropy:.4f}")

    
    def process_frame(self, frame):
        return super().process_frame(frame)
    
    
    def sample_trajectory(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        # Generate first frame
        action = 0
        obs, _, _ = self.game_state.frame_step(action)
        obs = self.process_frame(obs)
        obs = np.stack((obs, obs, obs, obs), axis=2).transpose(2, 0, 1)
        
        replay_buffer = Buffer(self.config.batch_size, self.config.input_shape, self.config.actions, self.config.gamma, self.config.lam)
        
        while True:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.config.device)
            action_dist = self.policy_net(obs_tensor)
            value = self.value_net(obs_tensor).item()
            
            action_dist = torch.distributions.Categorical(action_dist)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            action = action.item()
            obs_next_single, reward, terminal = self.game_state.frame_step(action)
            # for _ in range(1):
            #     _, _, _ = self.game_state.frame_step(0)
            obs_next_single = self.process_frame(obs_next_single)
            obs_next = np.append(obs[1:], obs_next_single.reshape(1, obs_next_single.shape[0], obs_next_single.shape[1]), axis=0)
            
            if replay_buffer.cur_idx < replay_buffer.max_size:
                replay_buffer.add(obs, action, reward, value, log_prob)
            else:
                return replay_buffer.get(replay_buffer.start_idx)
            obs = obs_next
            
            if terminal:
                replay_buffer.end_trajectory()
    
    
class Buffer:
    """
    Buffer for policy-based methods. Values and advantages are calculated.
    Ref: 
    """
    def __init__(self, size: int, obs_shape: tuple, num_actions: int, gamma: float=0.99, lam: float=0.95):
        self.obs = np.zeros((size, *obs_shape))
        self.actions = np.zeros(size)
        self.advantages = np.zeros(size)
        self.rewards = np.zeros(size)
        self.returns = np.zeros(size)
        self.values = np.zeros(size)
        self.logps = np.zeros(size)
        self.gamma, self.lam = gamma, lam
        self.cur_idx, self.start_idx, self.max_size = 0, 0, size
        self.obs_shape = obs_shape
        self.num_actions = num_actions
    
    
    def add(self, ob: np.ndarray, action: int, reward: float, value: float, logp: float):
        if self.cur_idx >= self.max_size:
            raise Exception("Buffer is full!")
        self.obs[self.cur_idx] = ob
        self.actions[self.cur_idx] = action
        self.rewards[self.cur_idx] = reward
        self.values[self.cur_idx] = value
        self.logps[self.cur_idx] = logp
        self.cur_idx += 1
    
    
    @staticmethod
    def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
        
    
    def end_trajectory(self, last_val: float=0.) -> None:
        rewards = np.append(self.rewards[self.start_idx:self.cur_idx], last_val)
        values = np.append(self.values[self.start_idx:self.cur_idx], last_val)
        
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantages[self.start_idx:self.cur_idx] = self.discount_cumsum(deltas, self.gamma * self.lam)
        
        self.returns[self.start_idx:self.cur_idx] = self.discount_cumsum(rewards, self.gamma)[:-1]
        
        self.start_idx = self.cur_idx
    
    
    def get(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        # Normalize advantages
        self.obs, self.actions, self.returns, self.advantages, self.logps, self.rewards = self.obs[:idx], self.actions[:idx], self.returns[:idx], self.advantages[:idx], self.logps[:idx], self.rewards[:idx]
        adv_mean = np.mean(self.advantages)
        adv_std = np.std(self.advantages)
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-6)
        return self.obs, self.actions, self.returns, self.advantages, self.logps, self.rewards.sum()