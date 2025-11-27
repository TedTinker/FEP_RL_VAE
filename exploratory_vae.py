from random import randint

import torch

from general_FEP_RL.agent import Agent

from encoders.encode_image import Encode_Image
from decoders.decode_image import Decode_Image
from encoders.encode_number import Encode_Number
from decoders.decode_number import Decode_Number

from utils import add_to_epoch_dict, plot_complete_epoch_dict
from get_data import plot_images, get_labeled_digits



observation_dict = {
    "see_number" : {
        "encoder" : Encode_Number,
        "decoder" : Decode_Number,
        "accuracy_scalar" : .1,                               
        "complexity_scalar" : .01,      
        "beta" : .01,                      
        "eta" : 0,
        },
    "see_image" : {
        "encoder" : Encode_Image,
        "decoder" : Decode_Image,
        "accuracy_scalar" : 1,                               
        "complexity_scalar" : .001,      
        "beta" : .01,                      
        "eta" : 1,
        },
    }

action_dict = {
    "make_number" : {
        "encoder" : Encode_Number,
        "decoder" : Decode_Number,
        "target_entropy" : -1,
        "alpha_normal" : .1
        },
    }



vae_agent = Agent(
    hidden_state_size = 256,
    observation_dict = observation_dict,       
    action_dict = action_dict,            
    number_of_critics = 1, 
    tau = .25,
    lr = .001,
    weight_decay = .00001,
    gamma = .99,
    capacity = 16, 
    max_steps = 26)


                    
epochs = 2000
episodes_per_epoch = 16
batch_size = 16
steps = 25
    
complete_epoch_dict = {}

for e in range(epochs): 
    print(f"Epoch {e}")
    
    print("Episode", end = " ")
    for episode in range(episodes_per_epoch):
        print(f"{episode}", end = " ")
        labeled_digits = get_labeled_digits()
        current_digit = randint(0, 9)
                
        vae_agent.begin()
                
        obs_list = []
        step_dict_list = []
                
        reward_list = []
                    
        for step in range(steps):            
            current_digit_tensor = torch.zeros([1, 1, 10])
            current_digit_tensor[:, :, current_digit] = 1
                        
            obs = {
                "see_number" : current_digit_tensor,
                "see_image" : labeled_digits[current_digit].unsqueeze(0).unsqueeze(0),
                }
            step_dict = vae_agent.step_in_episode(obs)
            
            reward = 0
            
            obs_list.append(obs)
            reward_list.append(reward)
            
            current_digit = torch.argmax(step_dict["action"]["make_number"]).item()
            
            if(step != steps-1):
                step_dict_list.append(step_dict)
            
        current_digit_tensor = torch.zeros([1, 1, 10])
        current_digit_tensor[:, :, current_digit] = 1
        final_obs = {
            "see_number" : current_digit_tensor,
            "see_image" : labeled_digits[current_digit].unsqueeze(0).unsqueeze(0)
            }
        step_dict = vae_agent.step_in_episode(final_obs)
        obs_list.append(final_obs)
        step_dict_list.append(step_dict)
        
        
        
        for i in range(len(reward_list)):
            current_digit_tensor = torch.zeros([1, 1, 10])
            current_digit = torch.argmax(step_dict_list[i]["action"]["make_number"]).item()
            current_digit_tensor[:, :, current_digit] = 1
            step_dict_list[i]["action"]["make_number"] = current_digit_tensor
            
            vae_agent.buffer.push(
                obs_list[i], 
                step_dict_list[i]["action"], 
                reward_list[i], 
                obs_list[i+1], 
                done = i == len(reward_list)-1)
                    
        
    digits = [obs["see_image"].squeeze(0).squeeze(0) for obs in obs_list]
    plot_images(digits, title = "REAL NUMBERS")
            
    x = torch.cat([step_dict["pred_obs_q"]["see_image"] for step_dict in step_dict_list], dim = 1)
    x = torch.cat([torch.ones_like(x[:,0].unsqueeze(0)), x], dim = 1)
    plot_images(x.squeeze(0), title = "PREDICTED NUMBERS")
        
    epoch_dict = vae_agent.epoch(batch_size = batch_size)
    add_to_epoch_dict(complete_epoch_dict, epoch_dict)
    #print_complete_epoch_dict(complete_epoch_dict)
    plot_complete_epoch_dict(complete_epoch_dict)



