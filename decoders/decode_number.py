import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity

from general_FEP_RL.utils_torch import init_weights, mu_std



# Decode Number (dn).
class Decode_Number(nn.Module):
    def __init__(
            self, 
            hidden_state_size, 
            encoded_action_size = 0, 
            entropy = False, 
            arg_dict = {"number_of_digits" : 10},
            verbose = False):
        super(Decode_Number, self).__init__()
                        
        self.example_input = torch.zeros(32, 16, hidden_state_size + encoded_action_size)
        if(verbose): 
            print("\nDN Start:", self.example_input.shape)

        example = self.example_input
        
        mu = nn.Sequential(
            nn.Linear(
                in_features = hidden_state_size,
                out_features = arg_dict["number_of_digits"]))
        
        self.mu_std = mu_std(mu, entropy = entropy)
        
        self.example_output, example_log_prob = self.mu_std(example)
        if(verbose): 
            print("\toutput:", self.example_output.shape)
            print("\tlog_prob:", example_log_prob.shape)
        
        self.apply(init_weights)
        
        
        
    def forward(self, hidden_state):
        output, log_prob = self.mu_std(hidden_state)
        output = F.softmax(output, dim = -1)
        return(output, log_prob)
    
    
    
    @staticmethod
    def loss_func(true_values, predicted_values):
        loss_value = F.mse_loss(predicted_values, true_values, reduction = "none")
        return loss_value
    
    
    
# Let's check it out!
if(__name__ == "__main__"):
    dn = Decode_Number(hidden_state_size = 128)
    print("\n\n")
    print(dn)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(dn, dn.example_input.shape))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    

    dn = Decode_Number(hidden_state_size = 128, entropy = True)
    print("\n\n")
    print(dn)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(dn, dn.example_input.shape))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    