from random import randint

import torch

from general_FEP_RL.agent import Agent
from encoders.encode_image import Encode_Image
from decoders.decode_image import Decode_Image
from encoders.encode_number import Encode_Number
from decoders.decode_number import Decode_Number

from utils import add_to_epoch_dict, plot_complete_epoch_dict
from get_data import plot_images, get_labeled_digits



folder = r"C:\Users\Ted\OneDrive\Desktop\FEP_RL_VAE\RSSM_without_exploration"



number_of_digits = 3



observation_dict = {
    "see_image" : {
        "encoder" : Encode_Image,
        "encoder_arg_dict" : {
            "encode_size" : 256,
            "zp_zq_sizes" : [256]},
        "decoder" : Decode_Image,
        "decoder_arg_dict" : {},
        "accuracy_scalar" : 1,                               
        "beta" : .001,                      
        "eta_before_clamp" : 100,
        "eta" : 0,
        },
    "see_number" : {
        "encoder" : Encode_Number,
        "encoder_arg_dict" : {
            "number_of_digits" : number_of_digits,
            "encode_size" : 16,
            "zp_zq_sizes" : [16]},
        "decoder" : Decode_Number,
        "decoder_arg_dict" : {
            "number_of_digits" : number_of_digits},
        "accuracy_scalar" : 1,                               
        "beta" : .001,      
        "eta_before_clamp" : 100,
        "eta" : 0,
        },
    }

# This actor/critic doesn't actually do anything.
action_dict = {
    "make_number" : {
        "encoder" : Encode_Number,
        "encoder_arg_dict" : {
            "number_of_digits" : number_of_digits,
            "encode_size" : 16,
            "zp_zq_sizes" : [16]},
        "decoder" : Decode_Number,
        "decoder_arg_dict" : {
            "number_of_digits" : number_of_digits},
        "target_entropy" : -1,
        "alpha_normal" : .1,
        "delta" : 0
        },
    }



# Still need: beta, eta_before_clamp, and eta for higher layers.
vae_agent = Agent(
    observation_dict = observation_dict,       
    action_dict = action_dict,  
    hidden_state_sizes = [1024, 128, 128],
    time_scales = [1, 2, 3],         
    beta = [1, 1],
    eta_before_clamp = [1, 1],
    eta = [1, 1],
    number_of_critics = 1, 
    tau = .99,
    lr = .001,
    weight_decay = .00001,
    gamma = .99,
    capacity = 32, 
    max_steps = 25)



vae_agent.world_model.summary()


                    
epochs = 5000
episodes_per_epoch = 16
steps_per_episode = 24
batch_size = 32


    
complete_epoch_dict = {}



for e in range(epochs): 
    print(f"\nEpoch {e}")
    
    print("Episode", end = " ")
    for episode in range(episodes_per_epoch):
        print(f"{episode}", end = " ")
        labeled_digits = get_labeled_digits()
        current_digit = 0
                
        vae_agent.begin()
                
        obs_list = []
        step_dict_list = []
                
        reward_list = []
                    
        for step in range(steps_per_episode):     
            current_digit_tensor = torch.zeros([1, 1, number_of_digits])
            current_digit_tensor[:, :, current_digit] = 1
            
            image = labeled_digits[current_digit].unsqueeze(0).unsqueeze(0).clone().detach()
            if(step == 0):
                image *= 0
                        
            obs = {
                "see_number" : current_digit_tensor,
                "see_image" : image,
                }
            step_dict = vae_agent.step_in_episode(obs)
            
            reward = 0
            
            obs_list.append(obs)
            reward_list.append(reward)
            
            current_digit = torch.argmax(step_dict["action"]["make_number"]).item()
            
            step_dict_list.append(step_dict)
            
        current_digit_tensor = torch.zeros([1, 1, number_of_digits])
        current_digit_tensor[:, :, current_digit] = 1
        final_obs = {
            "see_number" : current_digit_tensor,
            "see_image" : labeled_digits[current_digit].unsqueeze(0).unsqueeze(0),
            }
        step_dict = vae_agent.step_in_episode(final_obs)
        obs_list.append(final_obs)
        step_dict_list.append(step_dict)
        
        
        
        for i in range(len(reward_list)):
            vae_agent.buffer.push(
                observation_dict = obs_list[i], 
                action_dict = step_dict_list[i]["action"], 
                reward = reward_list[i], 
                next_observation_dict = obs_list[i+1], 
                done = i == len(reward_list)-1,
                best_action_dict = None)
                    
            
    
    epoch_dict = vae_agent.epoch(batch_size = batch_size)
    add_to_epoch_dict(complete_epoch_dict, epoch_dict)
    if(e % 10 == 0):
        plot_complete_epoch_dict(
            complete_epoch_dict, 
            folder = folder, 
            epoch = e)
    
        digits = [obs["see_image"].squeeze(0).squeeze(0) for obs in obs_list]
        plot_images(
            digits, 
            title = "REAL NUMBERS", 
            name=f"{e}", 
            folder=folder+"/real")
                
        x = torch.cat([step_dict["pred_obs_q"]["see_image"] for step_dict in step_dict_list], dim = 1)
        x = torch.cat([torch.ones_like(x[:,0].unsqueeze(0)), x], dim = 1)
        plot_images(
            x.squeeze(0)[:-1], 
            title = "PREDICTED NUMBERS", 
            name=f"{e}", 
            folder=folder+"/predicted")
        
