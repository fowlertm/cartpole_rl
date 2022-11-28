import numpy as np
import optuna
import matplotlib.pyplot as plt
from cartpole import train_cart_rl 
from copy import deepcopy
import datetime
import pickle as pkl



def objective(trial):
    
    config = {}

    # Network architecture
    config['net_arch'] = [dict(pi=[100,100,100,100], vf=[100,100,100,100])]

    # Learning parameters
    config['gamma'] = 0.999

    ###
    config['n_steps'] = trial.suggest_categorical('n_steps', [250, 500, 750])
    config['n_epochs'] = trial.suggest_categorical('n_epochs', [2, 5, 10])
    config['batch_size'] = trial.suggest_categorical('batch_size', [25, 50, 100])
    config['gamma'] = trial.suggest_categorical("gamma", [0.99, 0.999, 0.9999])
    config['ent_coef'] = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    config['clip_range'] = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    config['gae_lambda'] = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.95, 0.99, 1.0])
    config['max_grad_norm'] = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.7,  0.9, 1, 5])
    config['vf_coef'] = trial.suggest_uniform("vf_coef", 0, 1)
    config['state_type'] = trial.suggest_categorical('state_type',['p', 'pv', 'pva'])
    
    mean_reward, std_reward = train_cart_rl(config)
    print('trial: '+str(trial.number)+' mean reward: '+str(mean_reward))
    
    # Optuna does minimization so we we need negative sign
    return -mean_reward

def main():
    study = optuna.create_study()
    study.optimize(objective, n_trials=5)

    print()
    print(study.best_params)
    print()

if __name__ == "__main__":
    main()