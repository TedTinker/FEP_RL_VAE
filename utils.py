import os 

os.chdir(r"C:\Users\Ted\OneDrive\Desktop\FEP_RL_VAE")

print(os.getcwd())

import matplotlib.pyplot as plt



def print_epoch_dict(epoch_dict):
    print("\nprinting")
    for key, value in epoch_dict.items():
        print(f"{key}: {type(value)}")
        if(type(value) == float):
            print(f"\t{value}")
        if(type(value) == list):
            print(f"\t{value}")
        if(type(value) == dict):
            for k, v in value.items():
                print(f"\t{k}:")
                print(f"\t\t{v}")
                
                
                
def add_to_epoch_dict(complete_epoch_dict, epoch_dict):
    for key, value in epoch_dict.items():
                
        if(type(value) == float):
            if(not key in complete_epoch_dict):
                complete_epoch_dict[key] = [] 
            complete_epoch_dict[key].append(value)

        if(type(value) == list):
            if(not key in complete_epoch_dict):
                complete_epoch_dict[key] = []
                for i, v in enumerate(value):
                    complete_epoch_dict[key].append([])
            for i, v in enumerate(value):
                complete_epoch_dict[key][i].append(v) 
            
        if(type(value) == dict):
            if(not key in complete_epoch_dict):
                complete_epoch_dict[key] = {}
            for k, v in value.items():
                if(not k in complete_epoch_dict[key]):
                    complete_epoch_dict[key][k] = []
                complete_epoch_dict[key][k].append(v)
                
            
                    
def print_complete_epoch_dict(complete_epoch_dict):
    for key, value in complete_epoch_dict.items():
        print(f"{key}: {type(value)}")

        if(type(value) == list):
            if(type(value[0]) == list):
                for v in value:
                    print(f"\t{len(v)}")
            else:
                print(f"\t{len(value)}")
                    
        if(type(value) == dict):
            for k, v in value.items():
                print(f"\t{k}")
                print(f"\t\t{len(complete_epoch_dict[key][k])}")
                
                

def plot_complete_epoch_dict(complete_epoch_dict):
        
    plt.figure(figsize=(6, 6))
    for key, value in complete_epoch_dict["accuracy_losses"].items():
        plt.plot(value, label=f"accuracy loss {key}")
    for key, value in complete_epoch_dict["complexity_losses"].items():
        plt.plot(value, label=f"complexity loss {key}")
    plt.title(f"Losses for Accuracy and Complexity over epochs")
    plt.xlabel("Epoch")
    plt.ylabel(key)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(6, 6))
    plt.plot(complete_epoch_dict["total_reward"], label="total")
    plt.plot(complete_epoch_dict["reward"], label="reward")
    for key, value in complete_epoch_dict["curiosities"].items():
        plt.plot(value, label=f"curiosity {key}")
    plt.title(f"Rewards over epochs")
    plt.xlabel("Epoch")
    plt.ylabel(key)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(6, 6))
    plt.plot(complete_epoch_dict["actor_loss"], label="actor loss")
    for key, alpha_entropy in complete_epoch_dict["alpha_entropies"].items():
        plt.plot(alpha_entropy, label=f"alpha entropy {key}")
    for key, alpha_normal_entropy in complete_epoch_dict["alpha_normal_entropies"].items():
        plt.plot(alpha_normal_entropy, label=f"alpha normal entropy {key}")
    for key, total_entropy in complete_epoch_dict["total_entropies"].items():
        plt.plot(total_entropy, label=f"total entropy {key}")
    plt.title(f"Actor loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel(key)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(6, 6))
    for i, critic_loss in enumerate(complete_epoch_dict["critic_losses"]):
        plt.plot(critic_loss, label=f"critic {i} loss")
    plt.title(f"Critic loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel(key)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
     