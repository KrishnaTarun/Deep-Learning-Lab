import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import numpy as np
from datasets.bmnist import bmnist
from scipy.stats import norm

# np.random.seed(42)
torch.manual_seed(42)
# Initialize the device which to run the model on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim=500, z_dim=20):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear11 = nn.Linear(hidden_dim, z_dim)
        self.linear12 = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        hidden = self.relu(self.linear1(input))
        mean, std = self.linear11(hidden), self.softplus(self.linear12(hidden))
        
        return mean, std


class Decoder(nn.Module):

    def __init__(self, output_dim, hidden_dim=500, z_dim=20):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.linear1 = nn.Linear(z_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        y = self.linear1(input)
        y = self.relu(y)
        y = self.linear2(y)
        y = self.sigmoid(y)
         
        return y


class VAE(nn.Module):

    def __init__(self, input_dim, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
            

        self.encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.decoder = Decoder(input_dim, hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encostd
        negative average elbo for the given batch.
        """
        mu, std = self.encoder(input)
        z = self.reparameterize(mu, std)
        y = self.decoder(z)

        return mu, std, y 

       

    def reparameterize(self, mu, std):
        
        self.eps = torch.randn_like(std)
        z_sample = self.eps.mul(std).add_(mu)
        return z_sample
    

    # def sample(self, n_samples):
        
    #     """
    #     Sample n_samples from the model. Return both the sampled images
    #     (from bernoulli) and the means for these bernoullis (as these are
    #     used to plot the data manifold).
    #     """
    #     sampled_ims, im_means = None, None
    #     raise NotImplementedError()

    #     return sampled_ims, im_means


# returns reconstuction error
def log_bernoulli_loss(batch_data, x_recon):
    re_loss = batch_data*torch.log(x_recon+1e-8) + (1-batch_data)*torch.log(1-x_recon+1e-8)
    re_loss = torch.sum(re_loss, dim=1)

    return re_loss


# this returns negative of KL as in ELBO
def  kl_loss(mu, std):
    
    # sig_2 = torch.exp(self.logvar)
    kl = 0.5*(1+ torch.log(std**2) - mu.pow(2) - std.pow(2))

    kl = torch.sum(kl, dim=1)

    return kl

def loss_criterion(batch_data, mu, std, x_recon):
    l1 = log_bernoulli_loss(batch_data, x_recon) 
    l2 = kl_loss(mu, std)

    return -1*torch.sum(l1+l2)


def epoch_iter(epoch, model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    
    average_negative_elbo = 0
    for batch_idx, batch_data in enumerate(data):
        batch_data = batch_data.squeeze(1)
        batch_data = batch_data.view(batch_data.size()[0], -1).to(device)
        
        if model.training:
            model.zero_grad()
            mu, std, x_recon = model(batch_data)
            
            # this is basically loss 
            loss = loss_criterion(batch_data, mu, std, x_recon)
            loss.backward()    
            optimizer.step()

            average_negative_elbo+=loss.item()

            # save samples during training
            # --------------------------------------------------------------------
            #  Add functionality to plot samples from model during training.
            #  You can use the make_grid functioanlity that is already imported.
            # --------------------------------------------------------------------
            if epoch % 5==0 and batch_idx==0:
                model.eval()
                with torch.no_grad():
                    z = torch.randn((100, ARGS.zdim)).to(device)
                    plt_data =  model.decoder(z.float())
                    comparison = plt_data.view(plt_data.size(0),1,28,28)
                    save_image(comparison.cpu(), 'results/vae_reconstruction_' + str(epoch) + '.png', nrow=10)
                    print("Done")
                model.train()
            
            if batch_idx % ARGS.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(batch_data), len(data.dataset),
                100. * batch_idx / len(data), loss.item()/len(batch_data)))
        else:
            with torch.no_grad():
                mu, std, x_recon = model(batch_data)
                                                
                loss = loss_criterion(batch_data, mu, std, x_recon) 
                average_negative_elbo+=loss.item()
    
    if model.training:
        print('====> Epoch: {} Average Training loss: {:.4f}'.format(
            epoch, average_negative_elbo / len(data.dataset)))
    else:
        print('====> Epoch: {} Average Validation loss: {:.4f}'.format(
            epoch, average_negative_elbo / len(data.dataset)))
    
    average_epoch_elbo= average_negative_elbo/len(data.dataset)
  

    
    # NOTE: return lower bound which is -ve of loss i.e +ve
    return -1*average_epoch_elbo


def run_epoch(epoch, model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(epoch, model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(epoch,model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.grid(True)
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)

def vizManifold(model, n_points=20):
    
    # Display a 2D manifold of the digits
    n = 20  # figure with 20x20 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))

    # Construct grid of latent variable values
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    # decode for each square in the grid
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = torch.from_numpy((np.array([[xi, yi]]))).to(device)
            x_decoded = model.decoder(z_sample.float())
            # print(x_decoded.size())
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray')
    plt.show() 
    plt.close()

def main():
    
    
    
    data = bmnist()[:2]  # ignore test split

    model = VAE(ARGS.input_dim, hidden_dim=ARGS.hidden_dim, z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())
    model.to(device)

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(epoch, model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print('-------------------------ELBO--------------------------------------')
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")
        print('----------------------------------------------------------------')

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')

    #generate last reconstruction
    model.eval()
    with torch.no_grad():
        z = torch.randn((100, ARGS.zdim)).to(device)
        plt_data =  model.decoder(z.float())
        comparison = plt_data.view(plt_data.size(0),1,28,28)
        save_image(comparison.cpu(), 'results/vae_reconstruction_' + 'final' + '.png', nrow=10)
        print("Done")    

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------
    if ARGS.zdim==2:
        print("plot manifold")
        model.eval()
        with torch.no_grad():
            vizManifold(model)

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--hidden_dim', default=500, type=int,
                        help='dimensionality of hidden space')
    parser.add_argument('--input_dim', default=784, type=int,
                        help='dimensionality of input data')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
    ARGS = parser.parse_args()

    main()
