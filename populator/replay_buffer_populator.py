import pyRDDLGym
from pyRDDLGym_jax.core.planner import JaxBackpropPlanner, JaxOfflineController, load_config
from stable_baselines3.common.buffers import DictReplayBuffer
import numpy as np
from gymnasium import spaces
import os
import pickle

def shuffle_replay_buffer(buffer):
    """
    Takes a replay buffer and shuffles in place
    """
    size = buffer.size()
    if size == 0:
        return
    
    # Create a random permutation of indices
    indices = np.random.permutation(size)
    
    for key in buffer.observations.keys():
        buffer.observations[key][:size] = buffer.observations[key][indices]
        buffer.next_observations[key][:size] = buffer.next_observations[key][indices]
    buffer.actions[:size] = buffer.actions[indices]
    buffer.rewards[:size] = buffer.rewards[indices]
    buffer.dones[:size] = buffer.dones[indices]
    buffer.timeouts[:size] = buffer.timeouts[indices]

def populate(environment, instance, config_file, save_directory, obs_space, iterations=1):
    # Initialization
    env = pyRDDLGym.make(environment, instance, vectorized=True)
    planner_args, _, train_args = load_config(config_file)
    planner = JaxBackpropPlanner(rddl=env.model, **planner_args)
    controller = JaxOfflineController(planner, **train_args)

    buffer_size = 20000000 # Size of replay buffer that we will store

    # Action space and observation space needs to be declared. For MountainCars, the observation space needs to be
    # converted into a dict observation space. But action can be left alone.
    action_space = env.action_space['action']

    replay_buffer = DictReplayBuffer(
        buffer_size,
        obs_space,
        action_space,
        device="auto",
        n_envs=1,
        optimize_memory_usage=False,
        handle_timeout_termination=True
    )
    episodes_ended = 0
    episode_lengths = []  # List to store episode lengths

    # Collect X number of episodes, where X = number of iterations specified in function params
    for episode in range(iterations):
        if episode % 50 == 0:
            print(f"Episode {episode}")
        
        state, _ = env.reset()
        done = False
        episode_length = 0

        while not done:
            action = controller.sample_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Extra handling for when episodes end, for logging
            done = terminated or truncated
            if terminated: 
                episodes_ended += 1
            episode_length += 1

            # Adding the transition to the replay buffer
            obs = {
                'pos': np.array([state['pos']], dtype=np.float64),
                'vel': np.array([state['vel']], dtype=np.float64)
            }
            next_obs = {
                'pos': np.array([next_state['pos']], dtype=np.float64),
                'vel': np.array([next_state['vel']], dtype=np.float64)
            }
            act = action['action']
            infos = [info]
            
            replay_buffer.add(
                obs,
                next_obs,
                act,
                reward,
                done,
                infos
            )
            
            state = next_state

        # RESET THE CONTROLLER
        controller.reset()
        episode_lengths.append(episode_length) # log down the episod length

    env.close()

    # Calculate average episode length for logging
    avg_episode_length = np.mean(episode_lengths)
    print(f"Average episode length: {avg_episode_length:.2f}")

    print("Shuffling replay buffer...")
    shuffle_replay_buffer(replay_buffer)

    buffer_data = {
        'observations': {key: replay_buffer.observations[key][:replay_buffer.size()] for key in replay_buffer.observations.keys()},
        'next_observations': {key: replay_buffer.next_observations[key][:replay_buffer.size()] for key in replay_buffer.next_observations.keys()},
        'actions': replay_buffer.actions[:replay_buffer.size()],
        'rewards': replay_buffer.rewards[:replay_buffer.size()],
        'dones': replay_buffer.dones[:replay_buffer.size()],
        'timeouts': replay_buffer.timeouts[:replay_buffer.size()],
        'pos': replay_buffer.pos,
        'full': replay_buffer.full,
        'observation_space': observation_space,
        'action_space': action_space,
        'buffer_size': buffer_size,
        'n_envs': 1,
        'optimize_memory_usage': False,
        'handle_timeout_termination': True,
        'episode_lengths': episode_lengths,
        'avg_episode_length': avg_episode_length
    }

    # Save the buffer data
    os.makedirs(save_directory, exist_ok=True)
    with open(os.path.join(save_directory, f"replay_buffer_{environment}_{instance}_with_{replay_buffer.size()}_samples.pkl"), "wb") as f:
        pickle.dump(buffer_data, f)

    print(f"Buffer size: {replay_buffer.size()}")
    print(f"Data saved to directory: {save_directory}")

if __name__ == "__main__":
    # MAIN FUNCTION - only set up for Mountaincar discrete for now
    environment = "MountainCar_Discrete_gym"
    instance = "0"
    config_file = "populator/MountainCar_Discrete_gym_slp.cfg" #CFG file provided by Mike
    save_directory = "replay_buffer_data"
    observation_space = spaces.Dict({
        'pos': spaces.Box(low=-1.2, high=0.6, shape=(1,), dtype=np.float64),
        'vel': spaces.Box(low=-0.07, high=0.07, shape=(1,), dtype=np.float64)
    })
    populate(environment=environment, 
             instance=instance, 
             config_file=config_file, 
             save_directory=save_directory, 
             iterations=10,
             obs_space=observation_space)