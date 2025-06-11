'''
Taken from examples/run_stable_baselines.py
'''
import os
from stable_baselines3 import A2C, DDPG, SAC, PPO, TD3

import pyRDDLGym

from pyRDDLGym_rl.core.agent import StableBaselinesAgent
from pyRDDLGym_rl.core.env import SimplifiedActionRDDLEnv
from dual_buffer_dqn import DualBufferDQN

METHODS = {'a2c': A2C, 'ddpg': DDPG, 'dqn': DualBufferDQN, 'ppo': PPO, 'sac': SAC, 
           'td3': TD3}

def main(domain, 
         instance, 
         method, 
         second_replay_buffer_file,
         model_save_directory,
         sampling_ratio=0.5, # percentage of each batch that comes from JAX replay buffer
         steps=2000000,
         learning_rate=None,
         evaluate_after_testing=False
         ):
    
    # set up the environment
    env = pyRDDLGym.make(domain, instance, 
                         base_class=SimplifiedActionRDDLEnv,
                         enforce_action_constraints=True)

    # train the agent
    kwargs = {'verbose': 1}
    if learning_rate is not None:
        kwargs['learning_rate'] = learning_rate
    
    if method == 'dqn':
        kwargs['buffer_size_2'] = 2000000
        kwargs['sampling_ratio'] = sampling_ratio
    
    # Use MultiInputPolicy for DQN to stay compatible with Dictionary observation format
    model = METHODS[method]('MultiInputPolicy', env, buffer_size=2000000, **kwargs)
    
    if method == 'dqn':
        try:
            print("Loading JAXPlan replay buffers")
            model.load_second_replay_buffer(second_replay_buffer_file)
            print(f"Loaded replay buffer 2 with {model.replay_buffer_2.size()} transitions")
            
            # Initialize the model by running learn for 1 step
            print("Initializing model...")
            model.learn(total_timesteps=1)
        except FileNotFoundError:
            raise Exception("No existing replay buffers found, starting from scratch")
    # Continue training with environment interaction
    print("Starting environment interaction training...")
    model.learn(total_timesteps=steps)
    
    # Save the model and buffers
    os.makedirs(model_save_directory, exist_ok=True)
    model_save_path = os.path.join(model_save_directory, f"{domain}_{instance}_steps_{steps}_model")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    if method == 'dqn':
        model.save_replay_buffer(os.path.join(model_save_directory, f"{domain}_{instance}_steps_{steps}_replay_buffer"))
        print("Replay buffers saved")

    if evaluate_after_testing:
        agent = StableBaselinesAgent(model)
        agent.evaluate(env, episodes=1, verbose=False, render=True)
    
    env.close()
        
        
if __name__ == "__main__":
    # MAIN FUNCTION - also only set up for Mountaincar discrete for now
    environment = "MountainCar_Discrete_gym"
    instance = "0"
    # EDIT THIS LINE HERE TO CHANGE THE REPLAY BUFFER FILE
    second_replay_buffer = "replay_buffer_data/replay_buffer_MountainCar_Discrete_gym_0_with_615_samples.pkl"
    
    training_steps = 200_000
    main(domain=environment, 
         instance=instance, 
         method="dqn",
         second_replay_buffer_file=second_replay_buffer,
         model_save_directory="models",
         evaluate_after_testing=True,
         sampling_ratio=0.5,
         steps=training_steps)

