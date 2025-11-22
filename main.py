from general_FEP_RL.agent import Agent

from encoders.encode_image import Encode_Image
from encoders.encode_description import Encode_Description

from decoders.decode_image import Decode_Image
from decoders.decode_description import Decode_Description

from processor import variational_autoencoder



observation_dict = {
    "see_image" : {
        "encoder" : Encode_Image,
        "decoder" : Decode_Image,
        "accuracy_scalar" : 1,                               
        "complexity_scalar" : .01,      
        "beta" : 1,                      
        "eta" : 0,
        },
    }

action_dict = {
    "make_image" : {
        "encoder" : Encode_Image,
        "decoder" : Decode_Image,
        "target_entropy" : -1000,
        "alpha_normal" : .1
        }
    }



vae_agent = Agent(
    hidden_state_size = 512,
    observation_dict = observation_dict,       
    action_dict = action_dict,            
    number_of_critics = 5, 
    tau = .25,
    lr = .001,
    weight_decay = .00001,
    gamma = .99,
    capacity = 256, 
    max_steps = 11)



variational_autoencoder(vae_agent)