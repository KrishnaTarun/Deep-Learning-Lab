
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from a3_gan_template import Generator
import numpy as np
import os
import matplotlib.pyplot as plt

# Create output image directory
os.makedirs('interpolation', exist_ok=True)

# Initialize the device which to run the model on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Generator(100).to(device)
model.load_state_dict(torch.load("mnist_generator.pt"))


model.eval()
for j in range(10):
    a = torch.randn((1, 100)).to(device)
    b = torch.randn((1, 100)).to(device)

    batch_input = a
    for i, alpha in enumerate(np.linspace(start=0.0, stop=1.0, num=7)):
        
        z = (1-alpha) * a + alpha * b
        batch_input = torch.cat((batch_input,z.to(device) ), dim=0)

    batch_input = torch.cat((batch_input, b), dim=0)

    # print(batch_input)
    gen_img = model(batch_input)

    gen_img = gen_img.view(gen_img.size(0),1,28,28)
    save_image(gen_img.cpu(),'interpolation/{}_{}.png'.format("interpolated",j),
                        nrow=9, normalize=True)
