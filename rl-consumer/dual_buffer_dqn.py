from stable_baselines3.dqn import DQN
from stable_baselines3.common.buffers import DictReplayBuffer
import torch as th
import numpy as np
from typing import Optional, Dict, Any, Union
from stable_baselines3.common.type_aliases import GymEnv
from gymnasium import spaces
import pickle

class DualBufferDQN(DQN):
    def __init__(
        self,
        policy: Union[str, type[DQN]],
        env: Union[GymEnv, str],
        buffer_size_2: int = 1_000_000,  # Size of second buffer
        sampling_ratio: float = 0.5,  # Ratio of samples to take from buffer 1 vs buffer 2
        **kwargs
    ):
        super().__init__(policy, env, **kwargs)

        # Create second replay buffer
        self.replay_buffer_2 = DictReplayBuffer(
            buffer_size=buffer_size_2,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            optimize_memory_usage=False,
            handle_timeout_termination=True
        )

        self.sampling_ratio = sampling_ratio
        self.losses = []

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Calculate batch sizes for each buffer
            batch_size_2 = int(batch_size * self.sampling_ratio)
            batch_size_1 = batch_size - batch_size_2

            # Sample from both buffers
            replay_data_1 = self.replay_buffer.sample(batch_size_1, env=self._vec_normalize_env)
            replay_data_2 = self.replay_buffer_2.sample(batch_size_2, env=self._vec_normalize_env)

            # Combine the data
            replay_data = type('ReplayData', (), {
                'observations': {
                    key: th.cat([replay_data_1.observations[key], replay_data_2.observations[key]], dim=0)
                    for key in replay_data_1.observations.keys()
                },
                'next_observations': {
                    key: th.cat([replay_data_1.next_observations[key], replay_data_2.next_observations[key]], dim=0)
                    for key in replay_data_1.next_observations.keys()
                },
                'actions': th.cat([replay_data_1.actions, replay_data_2.actions], dim=0),
                'rewards': th.cat([replay_data_1.rewards, replay_data_2.rewards], dim=0),
                'dones': th.cat([replay_data_1.dones, replay_data_2.dones], dim=0)
            })

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = th.nn.functional.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.losses.append(np.mean(losses))

    def load_second_replay_buffer(self, path: str) -> None:
        """Load the second replay buffer from a file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.replay_buffer_2 = DictReplayBuffer(
                buffer_size=self.replay_buffer_2.buffer_size,
                observation_space=self.observation_space,
                action_space=self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=False,
                handle_timeout_termination=True
            )
            # Load dictionary observations
            for key in self.observation_space.spaces.keys():
                self.replay_buffer_2.observations[key] = data['observations'][key]
                self.replay_buffer_2.next_observations[key] = data['next_observations'][key]
            self.replay_buffer_2.actions = data['actions']
            self.replay_buffer_2.rewards = data['rewards']
            self.replay_buffer_2.dones = data['dones']
            self.replay_buffer_2.pos = data['pos']
            self.replay_buffer_2.full = data['full']

    def save_second_replay_buffer(self, path: str) -> None:
        """Save the second replay buffer to a file."""
        data = {
            'observations': self.replay_buffer_2.observations,
            'next_observations': self.replay_buffer_2.next_observations,
            'actions': self.replay_buffer_2.actions,
            'rewards': self.replay_buffer_2.rewards,
            'dones': self.replay_buffer_2.dones,
            'pos': self.replay_buffer_2.pos,
            'full': self.replay_buffer_2.full
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)