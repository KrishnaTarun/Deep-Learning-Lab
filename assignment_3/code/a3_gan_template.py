import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from tensorboardX import SummaryWriter

torch.manual_seed(42)

# Initialize the device which to run the model on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

writer = SummaryWriter()

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        self.l1 = nn.Linear(latent_dim, 128)
        #   LeakyReLU(0.2)
        self.leaky =  nn.LeakyReLU(0.2)
        #   Linear 128 -> 256
        self.l2 = nn.Linear(128, 256)
        #   Bnorm
        self.b1 = nn.BatchNorm1d(256) 
        #   LeakyReLU(0.2)
        # NOTE done
        #   Linear 256 -> 512
        self.l3 = nn.Linear(256, 512)
        #   Bnorm
        self.b2 = nn.BatchNorm1d(512)
        #   LeakyReLU(0.2)
        # NOTE done
        #   Linear 512 -> 1024
        self.l4 = nn.Linear(512, 1024)
        #   Bnorm
        self.b3 = nn.BatchNorm1d(1024)
        #   LeakyReLU(0.2)
        # NOTE done
        #   Linear 1024 -> 768
        self.l5 = nn.Linear(1024, 784)
        #   Output non-linearity
        self.tanh = nn.Tanh()

    def forward(self, z):
        # Generate images from z

        out = self.leaky(self.l1(z))
        out = self.l2(out)
        out = self.leaky(self.b1(out))
        
        out = self.l3(out)
        out = self.leaky(self.b2(out))

        out = self.l4(out)
        out= self.leaky(self.b3(out))

        out = self.tanh(self.l5(out))

        return out

class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        self.l1 = nn.Linear(input_dim, 512)
        #   LeakyReLU(0.2)
        self.leaky =  nn.LeakyReLU(0.2)
        #   Linear 512 -> 256
        self.l2 = nn.Linear(512, 256)
        #   LeakyReLU(0.2)
        # NOTE done
        #   Linear 256 -> 1
        self.l3 = nn.Linear(256, 1)
        #   Output non-linearity
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        # return discriminator score for img
        out = self.leaky(self.l1(img))
        out = self.leaky(self.l2(out))
        out = self.sigmoid(self.l3(out))

        return out


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, adversarial_loss):
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            imgs = imgs.view(imgs.size()[0], -1).to(device)

            # create labels for fake and real images
            true_label = torch.ones(imgs.size(0),).to(device)
            fake_label = torch.zeros(imgs.size(0),).to(device)

            # Train Generator
            # ---------------
            optimizer_G.zero_grad()

            z = torch.randn((imgs.shape[0], args.latent_dim)).to(device)

            gen_out = generator(z)
            loss_G  =   adversarial_loss(discriminator(gen_out).squeeze(-1), true_label)
            
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()
            
            # NOTE sample new noise
            z_new = torch.randn((imgs.shape[0], args.latent_dim)).to(device)
            
            true_loss = adversarial_loss(discriminator(imgs),true_label)
            gen_new = generator(z_new).detach()
            fake_loss = adversarial_loss(discriminator(gen_new).squeeze(-1), fake_label)

            loss_D = (true_loss+fake_loss)
            loss_D.backward()
            optimizer_D.step()

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                plt_data = gen_new[:64]
                plt_data = plt_data.view(plt_data.size(0),1,28,28)
                save_image(plt_data.cpu(),
                           'images/{}.png'.format(batches_done),
                           nrow=8, normalize=True)

            niter = epoch*len(dataloader)+i
            writer.add_scalar('D/Loss', loss_D.item(), niter)
            writer.add_scalar('G/Loss', loss_G.item(), niter)    
            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, args.n_epochs, i, len(dataloader),
                                                            loss_D.item(), loss_G.item()))
        
        
def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True)

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize models and optimizers
    generator = Generator(args.latent_dim).to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, adversarial_loss)

    
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    # You can save your generator here to re-use it to generate images for your
        # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
