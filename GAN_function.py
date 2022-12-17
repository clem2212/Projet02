import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from implementations import *

# n= 0    # 0 < E < 7.8  and 7.8 < E < 20 MeV

def GAN1_data(datapath):
    df = clean_df(build_df(datapath))
    df = df.loc[df['name_s'] == 0]
    df = get_cos_theta(df)
    
    df1 = df.loc[df['KinE(MeV)'] < 7.8]
    df2 = df.loc[df['KinE(MeV)'] >= 7.8]
    
    df1 = df1[['cos_theta','dE(MeV)']]
    df2 = df2[['cos_theta','dE(MeV)']]
    
    df1 = df1.to_numpy()
    df2 = df2.to_numpy()
    
    return df1, df2


#n= e-   # 0 < E < 1   and   1 < E < 20 MeV
def GAN2_data(datapath):
    df = clean_df(build_df(datapath))
    df = df.loc[df['name_s'] == 'e-']
    df = get_angles(df)
    
    df1 = df.loc[df['KinE(MeV)'] < 1.0]
    df2 = df.loc[df['KinE(MeV)'] >= 1.0]
    
    df1 = df1[['cos_theta','dE(MeV)', 'cos_phi', 'cos_psi', 'KinE(MeV)']]
    df2 = df2[['cos_theta','dE(MeV)', 'cos_phi', 'cos_psi', 'KinE(MeV)']]
    
    df1 = df1.to_numpy()
    df2 = df2.to_numpy()
    
    return df1, df2


#n= gamma   # 0 < E < 1   and   1 < E < 20 MeV
def GAN3_data(datapath):
    df = clean_df(build_df(datapath))
    df = df.loc[df['name_s'] == 'gamma']
    df = get_angles(df)
    
    df1 = df.loc[df['KinE(MeV)'] < 1.0]
    df2 = df.loc[df['KinE(MeV)'] >= 1.0]
    
    df1 = df1[['cos_theta','dE(MeV)', 'cos_phi', 'cos_psi', 'KinE(MeV)']]
    df2 = df2[['cos_theta','dE(MeV)', 'cos_phi', 'cos_psi', 'KinE(MeV)']]
    
    df1 = df1.to_numpy()
    df2 = df2.to_numpy()
    
    return df1, df2




def inf_data_gen(dataset=None, batch_size=None):
    """Python infinite iterator (called python-generator) of samples
    following certain distribution.

    Example Usage:
    data_generator = inf_data_gen(dataset='E_20.0.data', batch_size=64)
    sample = next(data_generator)
    """
    while True:
            data = dataset[np.random.choice(dataset.shape[0],\
                                            size=batch_size, replace=False), :]
            data = data.astype("float32")
            yield data
            
            
class GeneratorMLP(nn.Module):
    def __init__(self, dim_hidden=512, dim_out=3, noise_dim=100):
        super(GeneratorMLP, self).__init__()
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim, dim_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x):
        return self.net(x)

class DiscriminatorMLP(nn.Module):
    def __init__(self, dim_hidden=512, dim_gen_out=3):
        super(DiscriminatorMLP, self).__init__()
        self.dim_hidden = dim_hidden
        self.dim_gen_out = dim_gen_out
        self.net = nn.Sequential(
            nn.Linear(dim_gen_out, dim_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
    
    
    
def train_generator(gen, disc, loss=None, batch_size=64, device=torch.device("cpu")):
    """Updates the params of disc['model'] (once).

    :param gen: dictionary with key 'model' [torch.nn.Sequential] and
                'optim' [torch.optim]
    :param disc: dictionary with key 'model' [torch.nn.Sequential]
    :param loss: [torch.nn.<loss>]
    :param batch_size: [int]
    :param device: torch.device('cuda') or torch.device('cpu')
    :return: None
    """
    loss = loss or nn.BCELoss()  # Binary cross entropy
    labels = torch.ones(batch_size, 1, device=device)
    noise = torch.randn(batch_size, gen["model"].noise_dim, device=device)

    gen["model"].zero_grad()
    loss(disc["model"](gen["model"](noise)), labels).backward()
    gen["optim"].step()  # update params of the generator
    return



def train_discriminator(
    gen,
    disc,
    data_sampler,
    n_steps=1,
    loss=None,
    batch_size=64,
    device=torch.device("cpu"),
):
    """Updates the params of disc['model'] n_steps times.

    :param gen: dictionary with key 'model' [torch.nn.Sequential]
    :param disc: dictionary with key 'model' [torch.nn.Sequential] and
                 'optim' [torch.optim]
    :param data_sampler: [python generator (https://wiki.python.org/moin/Generators)]
    :param n_steps: [int]
    :param loss: [torch.nn.<loss>]
    :param batch_size: [int]
    :param device: torch.device('cuda') or torch.device('cpu')
    :return: None
    """
    real_labels = torch.ones(batch_size, 1, device=device)
    fake_labels = torch.zeros(batch_size, 1, device=device)
    loss = loss or nn.BCELoss()  # Binary cross entropy

    for _ in range(n_steps):
        disc["model"].zero_grad()
        #  1. Backprop - D on real: maximize log(D(x)) + log(1 - D(G(z)))
        real_samples = torch.tensor(next(data_sampler), device=device)
        loss(disc["model"](real_samples), real_labels).backward()

        #  2. Backprop - D on fake:
        noise = torch.randn(batch_size, gen["model"].noise_dim, device=device)
        loss(disc["model"](gen["model"](noise)), fake_labels).backward()

        #  3. Update the parameters  of the generator
        disc["optim"].step()
    return

def launch_GAN(g_lr=1e-4, d_lr=5e-4, batch_size=64, noise_dim=100, total_iterations=5000, criterion=nn.BCELoss(), data=None, g_model=None, d_model=None):
    
    device = torch.device("cpu")
    g_optim = torch.optim.Adam(g_model.parameters(), lr=g_lr, betas=(0.5, 0.999))
    d_optim = torch.optim.Adam(d_model.parameters(), lr=d_lr, betas=(0.5, 0.999))

    generator = {"model": g_model, "optim": g_optim}
    discriminator = {"model": d_model, "optim": d_optim}

    fixed_noise = torch.randn(300, noise_dim, device=device)
    data_generator = inf_data_gen(dataset=data, batch_size=batch_size)

    plot_frequency = total_iterations // 100
    fake_samples = []
    DONE=0
    for i in range(total_iterations):
        train_discriminator(
            generator,
            discriminator,
            data_sampler=data_generator,
            loss=criterion,
            batch_size=batch_size,
            device=device,
        )

        train_generator(
            generator, discriminator, loss=criterion, batch_size=batch_size, device=device
        )
        if (i%100 == 0) :
                sys.stdout.write(f"Finished {i:8} out of {total_iterations:8} {(100.0*i)/total_iterations:.2f} %\r"); sys.stdout.flush()

        if i % plot_frequency == 0 or (i + 1) == total_iterations:
            fake_samples.append(generator["model"](fixed_noise).cpu().detach().numpy())
    return fake_samples 
        
        
        
def train_GAN(g_lr=1e-4, d_lr=5e-4, batch_size=64, noise_dim=100, total_iterations=5000, criterion=nn.BCELoss(), data=None, g_model=None, d_model=None):
    
    device = torch.device("cpu")
    g_optim = torch.optim.Adam(g_model.parameters(), lr=g_lr, betas=(0.5, 0.999))
    d_optim = torch.optim.Adam(d_model.parameters(), lr=d_lr, betas=(0.5, 0.999))

    generator = {"model": g_model, "optim": g_optim}
    discriminator = {"model": d_model, "optim": d_optim}

    fixed_noise = torch.randn(300, noise_dim, device=device)
    data_generator = inf_data_gen(dataset=data, batch_size=batch_size)
    
    for i in range(total_iterations):
        train_discriminator(
            generator,
            discriminator,
            data_sampler=data_generator,
            loss=criterion,
            batch_size=batch_size,
            device=device,
        )

        train_generator(
            generator, discriminator, loss=criterion, batch_size=batch_size, device=device
        )
        if (i%100 == 0) :
                sys.stdout.write(f"Finished {i:8} out of {total_iterations:8} {(100.0*i)/total_iterations:.2f} %\r"); sys.stdout.flush()
    
    return generator["model"]



def Get_GAN_event(g_model=None,noise_dim=100):
    device = torch.device("cpu")
    fixed_noise = torch.randn(1, noise_dim, device=device)
    return g_model(fixed_noise).cpu().detach().numpy()
    

    
    
    
    


            