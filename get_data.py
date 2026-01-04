#%%

import os
import math
from keras.datasets import mnist 
import numpy as np
import random
from random import sample
import matplotlib.pyplot as plt

import torch



(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_y = np.eye(10)[train_y]
test_y  = np.eye(10)[test_y]

train_x = torch.from_numpy(train_x).reshape((train_x.shape[0], 28, 28, 1))
train_y = torch.from_numpy(train_y)
test_x  = torch.from_numpy(test_x).reshape((test_x.shape[0], 28, 28, 1))
test_y  = torch.from_numpy(test_y)

train_x = train_x/255
test_x  = test_x/255



def get_data(batch_size = 64, test = False):
    if(test): 
        x = test_x
        y = test_y
    else:     
        x = train_x
        y = train_y
    index = [i for i in range(len(x))]
    batch_index = sample(index, batch_size)
    x = x[batch_index]
    y = y[batch_index]
    return(x, y.float())




def get_repeating_digit_sequence_random_start(
    batch_size=64, 
    steps=20, 
    n_digits=3, 
    test=False
):
    x_data = test_x if test else train_x
    y_data = test_y if test else train_y

    assert n_digits <= 10, "n_digits must be <= 10"

    # 1. Choose one image per digit in the range [0, n_digits)
    digit_images = {}
    digit_labels = {}

    for d in range(n_digits):
        d_indices = (y_data.argmax(dim=1) == d).nonzero(as_tuple=True)[0]
        idx = d_indices[torch.randint(len(d_indices), (1,)).item()]
        digit_images[d] = x_data[idx]
        digit_labels[d] = y_data[idx]

    # --- NEW: random start and direction ----
    start_digit = random.randrange(n_digits)
    direction = random.choice([1, -1])   # forward or backward

    # 2. Create repeating digit pattern with wrap-around
    digit_pattern = [ (start_digit + direction * i) % n_digits 
                      for i in range(steps) ]

    # 3. Build batch
    x_batch = []
    y_batch = []

    for _ in range(batch_size):
        x_seq = [digit_images[d] for d in digit_pattern]
        y_seq = [digit_labels[d] for d in digit_pattern]

        x_batch.append(torch.stack(x_seq))      # (steps, 28, 28, 1)
        y_batch.append(torch.stack(y_seq))      # (steps, 10)

    x_batch = torch.stack(x_batch)              # (batch, steps, 28, 28, 1)
    y_batch = torch.stack(y_batch).float()      # (batch, steps, 10)

    return x_batch, y_batch



def get_labeled_digits(
    test=False
):
    x_data = test_x if test else train_x
    y_data = test_y if test else train_y
    
    labeled_digits = {} 
    
    for d in range(10):
        d_indices = (y_data.argmax(dim=1) == d).nonzero(as_tuple=True)[0]
        idx = d_indices[torch.randint(len(d_indices), (1,)).item()]
        labeled_digits[d] = x_data[idx]

    return labeled_digits



def plot_images(images, title, show=True, name="", folder=""):
    n_images = len(images)
    columns = math.ceil(math.sqrt(n_images))
    rows = math.ceil(n_images / columns)
    fig = plt.figure(figsize=(columns + 1, rows + 1))
    fig.suptitle(title)
    for i in range(1, rows * columns + 1):
        ax = fig.add_subplot(rows, columns, i)
        if i <= n_images:
            ax.imshow(images[i - 1], cmap="gray")
        ax.axis("off")
    if name:
        os.makedirs(f"{folder}", exist_ok=True)
        plt.savefig(f"{folder}/{name}.png")
    if show:
        plt.show()
    plt.close()



def get_display_data():
    images, digits = get_data(256, True)
    images_for_display = []
    digits_for_display = []
    for i in range(10):
        for j, d in enumerate(digits):
            if(d.argmax().item() == i):
                images_for_display.append(images[j].unsqueeze(0))
                digits_for_display.append(d.unsqueeze(0))
                break 
    return(
        torch.cat(images_for_display).cpu(), 
        torch.cat(digits_for_display))



if __name__ == "__main__":
    x, y = get_display_data()
    plot_images(x, "Real numbers", show = True) 
# %%