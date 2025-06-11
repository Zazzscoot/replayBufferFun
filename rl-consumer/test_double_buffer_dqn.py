'''In this example, the stable baselines package is used to train an RL agent.
    
The syntax for running this example is:

    python run_stable_baselines.py <domain> <instance> <method> [<steps>] [<learning_rate>]
    
where:
    <domain> is the name of a domain located in the /Examples directory
    <instance> is the instance number
    <method> is the algorithm to train (e.g. PPO, DQN etc.)
    <steps> is the number of trials to simulate for training
    <learning_rate> is the learning rate to use to train the agent
'''
import os
from stable_baselines3 import A2C, DDPG, SAC, PPO, TD3

import pyRDDLGym

from pyRDDLGym_rl.core.agent import StableBaselinesAgent
from pyRDDLGym_rl.core.env import SimplifiedActionRDDLEnv
from dual_buffer_dqn import DualBufferDQN

def main(domain, 
         instance,
         model_filename):
    # setup the environment
    env = pyRDDLGym.make(domain, instance, 
                         base_class=SimplifiedActionRDDLEnv,
                         enforce_action_constraints=True)
    
    # Load model
    model = DualBufferDQN.load(model_filename, env=env)
    agent = StableBaselinesAgent(model)

    # Test
    agent.evaluate(env, episodes=1, verbose=False, render=True)
    env.close()
        
        
if __name__ == "__main__":
    main(domain="MountainCar_Discrete_gym", 
         instance="0", 
         # MAKE SURE model_filename is correct! 
         model_filename="models/MountainCar_Discrete_gym_0_steps_2000000_model.zip")
