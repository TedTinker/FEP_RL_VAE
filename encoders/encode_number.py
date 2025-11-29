import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity

from general_FEP_RL.utils_torch import init_weights, model_start, model_end, Interpolate



# Encode Number (en).
class Encode_Number(nn.Module):
    def __init__(
            self, 
            arg_dict = {"number_of_digits" : 10},
            verbose = False):
        super(Encode_Number, self).__init__()
        
        self.out_features = 16
                
        self.example_input = torch.zeros(1, 1, arg_dict["number_of_digits"])
        if(verbose):
            print("\nEN Start:", self.example_input.shape)
        example = self.example_input
        example = example.argmax(dim = -1)
                
        self.a = nn.Sequential(
            nn.Embedding(
                num_embeddings = arg_dict["number_of_digits"], 
                embedding_dim = self.out_features),
            nn.PReLU())
                
        example = self.a(example)
        if(verbose): 
            print("\toutput:", example.shape)
        
        self.example_output = example
        if(verbose):
            print("EN End:")
            print("\toutput:", example.shape, "\n")
        
        self.apply(init_weights)
        
        
        
    def forward(self, number):
        number_index = number.argmax(dim=-1)
        output = self.a(number_index)
        return(output)
    
    
    
# Let's check it out!
if(__name__ == "__main__"):
    en = Encode_Number(verbose = True)
    print("\n\n")
    print(en)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(en, en.example_input.shape))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    
    
    example_dict = {
        "encoder" : en,
        "target_entropy" : 1,
        "accuracy_scaler" : 1,                               
        "complexity_scaler" : 1,                                 
        "eta" : 1                                   
        }
    