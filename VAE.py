import torch.optim as optim

from encoders.encode_image import Encode_Image
from decoders.decode_image import Decode_Image

from utils import add_to_epoch_dict, plot_complete_epoch_dict
from get_data import plot_images, get_data



encoder = Encode_Image()
decoder = Decode_Image(hidden_state_size = 128, entropy = True)

all_parameters = list(encoder.parameters()) + list(decoder.parameters())
vae_opt = optim.Adam(all_parameters, lr = .001, weight_decay = .0001) 



epochs = 2000
batch_size = 16
beta = .0001
    


for e in range(epochs): 
    print(f"Epoch {e}")

    x, y = get_data(batch_size = 16, test = False)
    x = x.unsqueeze(1)
                            
    encoded = encoder(x)
    decoded, log_prob = decoder(encoded)
    
    accuracy = decoder.loss_func(x, decoded).mean()
    complexity = - log_prob.mean() * beta
    loss = accuracy + complexity
    
    vae_opt.zero_grad()
    loss.backward()
    vae_opt.step()
    
    if(e % 100 == 0):
        print(loss.item())
        print(x.shape, decoded.shape)
        
        plot_images(x.squeeze(1), "REAL", show=True, name="", folder="")
        plot_images(decoded.squeeze(1).detach().numpy(), "DECODED", show=True, name="", folder="")