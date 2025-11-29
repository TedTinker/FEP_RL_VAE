import torch

from general_FEP_RL.agent import Agent
from encoders.encode_image import Encode_Image
from decoders.decode_image import Decode_Image
from encoders.encode_number import Encode_Number
from decoders.decode_number import Decode_Number

from utils import add_to_epoch_dict, plot_complete_epoch_dict
from get_data import plot_images, get_repeating_digit_sequence_random_start



number_of_digits = 4



observation_dict = {
    "see_image" : {
        "encoder" : Encode_Image,
        "decoder" : Decode_Image,
        "arg_dict" : {},
        "accuracy_scalar" : 1,                               
        "beta" : .001,                      
        "eta" : 0,
        },
    }

# This actor/critic doesn't actually do anything.
action_dict = {
    "make_number" : {
        "encoder" : Encode_Number,
        "decoder" : Decode_Number,
        "arg_dict" : {"number_of_digits" : number_of_digits},
        "target_entropy" : -1,
        "alpha_normal" : .1
        },
    }



vae_agent = Agent(
    hidden_state_size = 512,
    observation_dict = observation_dict,       
    action_dict = action_dict,            
    number_of_critics = 1, 
    tau = .99,
    lr = .002,
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
        x, y = get_repeating_digit_sequence_random_start(
            batch_size = 1, 
            steps = steps, 
            n_digits = number_of_digits,
            test = False)
                
        vae_agent.begin()
                
        obs_list = []
        step_dict_list = []
                
        reward_list = []
                    
        for step in range(steps):            
            obs = {
                "see_image" : x[:,step].unsqueeze(1),
                }
            step_dict = vae_agent.step_in_episode(obs)
            
            reward = 0
            
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
                    
        
        
    plot_images(x.squeeze(0), title = "REAL NUMBERS")
            
    x = torch.cat([step_dict["pred_obs_q"]["see_image"] for step_dict in step_dict_list], dim = 1)
    x = torch.cat([torch.ones_like(x[:,0].unsqueeze(0)), x[:, :-1]], dim = 1)
    plot_images(x.squeeze(0), title = "PREDICTED NUMBERS")
        
    epoch_dict = vae_agent.epoch(batch_size = batch_size)
    add_to_epoch_dict(complete_epoch_dict, epoch_dict)
    #print_complete_epoch_dict(complete_epoch_dict)
    plot_complete_epoch_dict(complete_epoch_dict)



