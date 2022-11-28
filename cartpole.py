import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

def train_cart_rl(config):
    
    train_env = gym.make('CartPole-v1')

    model = PPO(MlpPolicy, train_env, verbose=0, n_steps=config['n_steps'], n_epochs=config['n_epochs'],
                    batch_size=config['batch_size'], gamma=config['gamma'],
                    ent_coef=config['ent_coef'], clip_range=config['clip_range'],
                    gae_lambda=config['gae_lambda'], max_grad_norm=config['max_grad_norm'],
                    vf_coef=config['vf_coef'])

    # Use a separate environment for evaluation
    eval_env = gym.make('CartPole-v1')

    # Random Agent, before training
    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)

    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    # Train the agent for 10000 steps
    model.learn(total_timesteps=10000)

    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)

    return mean_reward, std_reward
    #print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":

    config = {}

    # PPO parameters
    config['n_steps'] = 250
    config['n_epochs'] = 10
    config['batch_size'] = 50
    config['ent_coef'] = 3.5e-6
    config['clip_range'] = .3
    config['gae_lambda'] = 0.9
    config['max_grad_norm'] = 0.7
    config['vf_coef'] = 0.67

    # Network architecture
    config['net_arch'] = [dict(pi=[100,100,100,100], vf=[100,100,100,100])]

    # Learning parameters
    config['gamma'] = 0.999
    # config['initial_lr'] = .0001
    # config['lr_schedule_type'] = 'StepLR'
    # config['learning_start_frac'] = .5
    # config['frac_to_decrease_lr'] = .015
    # config['lr_annealing_rate'] = .9
    # config['initial_lr'] = 1e-4
    # config['final_lr'] = 1e-6
    # config['power'] = 2.5
    # config['lr_schedule_type'] = 'Exponential'
    # config['start_decay'] = .9
    # config['initial_lr'] = 1e-4
    # config['lr_schedule_type'] = 'Constant'

    train_cart_rl(config)
