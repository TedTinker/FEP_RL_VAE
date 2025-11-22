
import torch

from utils import print_epoch_dict, add_to_epoch_dict, print_complete_epoch_dict, plot_complete_epoch_dict
from get_data import get_repeating_digit_sequence, plot_images
from general_FEP_RL.utils_torch import print_step_in_episode


                    
def variational_autoencoder(
        vae_agent, 
        epochs = 1000, 
        episodes_per_epoch = 32, 
        batch_size = 64, 
        steps = 10):
    
    complete_epoch_dict = {}
    
    for e in range(epochs):
        print(f"Epoch: {e}")
        
        for episode in range(episodes_per_epoch):
            
            x, y = get_repeating_digit_sequence(batch_size=1, steps=steps, test=False)
                    
            vae_agent.begin()
                    
            obs_list = []
            step_dict_list = []
                    
            reward_list = []
            
            prev_pred_image = 0 * x[:,0].unsqueeze(0)
            
            for step in range(steps):            
                obs = {
                    "see_image" : x[:,step].unsqueeze(1),
                    }
                step_dict = vae_agent.step_in_episode(obs)
                
                reward = -vae_agent.world_model.action_dict["make_image"]["decoder"].loss_func(
                    x[:,step].unsqueeze(1), step_dict["action"]["make_image"]).mean()
                
                obs_list.append(obs)
                reward_list.append(reward)
                
                if(step != steps-1):
                    step_dict_list.append(step_dict)
                
                            
                
            final_obs = {
                "see_image" : x[:,steps-1].unsqueeze(1)}
            step_dict = vae_agent.step_in_episode(final_obs)
            obs_list.append(final_obs)
            step_dict_list.append(step_dict)
            
            
            
            for i in range(len(reward_list)):
                vae_agent.buffer.push(
                    obs_list[i], 
                    step_dict_list[i]["action"], 
                    reward_list[i], 
                    obs_list[i+1], 
                    done = i == len(reward_list)-1)
                        
            
            
        print("real:", x.shape)
        plot_images(x.squeeze(0), title = "REAL NUMBERS")
        
        x = torch.cat([step_dict["action"]["make_image"] for step_dict in step_dict_list], dim = 1)
        x = torch.cat([torch.ones_like(x[:,0].unsqueeze(0)), x[:, :-1]], dim = 1)
        print("actor:", x.shape)
        plot_images(x.squeeze(0), title = "ACTOR'S NUMBERS")
                
        x = torch.cat([step_dict["pred_obs_q"]["see_image"] for step_dict in step_dict_list], dim = 1)
        x = torch.cat([torch.ones_like(x[:,0].unsqueeze(0)), x[:, :-1]], dim = 1)
        print("predictions:", x.shape)
        plot_images(x.squeeze(0), title = "PREDICTED NUMBERS")
            
        epoch_dict = vae_agent.epoch(batch_size = batch_size)
        add_to_epoch_dict(complete_epoch_dict, epoch_dict)
        #print_complete_epoch_dict(complete_epoch_dict)
        plot_complete_epoch_dict(complete_epoch_dict)

        
        
if __name__ == "__main__": 
    variational_autoencoder(None)