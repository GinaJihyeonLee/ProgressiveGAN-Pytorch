from config import config
import model
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import utils
from torchvision.utils import save_image, make_grid
import os
import dataloader as DL
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, config):
        self.resl = 2
        self.epoch = 0
        self.config=config
        self.lr=self.config.lr
        self.max_iter=[20000, 20000, 20000, 50000, 50000, 100000, 100000, 100000, 100000]
        self.G = model.Generator(config)
        self.D = model.Discriminator(config)
        print('Generator structure: ')
        print(self.G.main_layers, self.G.to_rgb_layers)
        print('Discriminator structure: ')
        print(self.D.from_rgb_layers, self.D.main_layers)
        self.G = DataParallel(self.G)
        self.D = DataParallel(self.D)
        if config.loss_type=='lsgan':
            self.gan_loss = nn.MSELoss()
        elif config.loss_type=='wgan-gp': 
            self.gan_loss = None
        self.device = torch.device("cuda")
        betas=(self.config.beta1, self.config.beta2)
        self.opt_g=Adam(self.G.parameters(), lr=self.lr, betas=betas, weight_decay=0.0)
        self.opt_d=Adam(self.D.parameters(), lr=self.lr, betas=betas, weight_decay=0.0)
        self.fixed_z = torch.rand(64,self.config.nz,1,1).to(self.device)
        self.writer = SummaryWriter(config.log_root)

    def train(self):
        for step in range(2, self.config.max_resl+1):
            loader, self.batchsize = DL.data_loader(self.config,self.resl)
            dataloader=iter(loader)
            self.real_label = torch.ones(self.batchsize).to(self.device)
            self.fake_label = torch.zeros(self.batchsize).to(self.device)
            alpha=0.0
            if step==2: alpha=1
            for iters in tqdm(range(0,self.max_iter[step-2])):
              self.G.to(self.device)
              self.D.to(self.device)
              self.G.zero_grad()
              self.D.zero_grad()
              try:
                  self.real_img=next(dataloader)
              except StopIteration:
                  dataloader = iter(loader)
                  self.real_img = next(dataloader)
              self.real_img = self.real_img.to(self.device)
              self.z = torch.rand(self.batchsize, self.config.nz).to(self.device)

              #update discriminator
              self.fake_img = self.G(self.z, alpha)
              self.real_pred = self.D(self.real_img, alpha)
              self.fake_pred = self.D(self.fake_img.detach(), alpha)
              if self.config.loss_type=='lsgan':
                  loss_d = self.gan_loss(self.real_pred, self.real_label)+self.gan_loss(self.fake_pred,self.fake_label)
              elif self.config.loss_type=='wgan-gp':
                  beta = torch.rand(self.batchsize, 1, 1, 1).to(self.device)
                  x_hat = (beta * self.real_img + (1 - beta) * self.fake_img).requires_grad_(True)
                  x_hat_out = self.D(x_hat, alpha)
                  loss_d = -self.real_pred.mean() + self.fake_pred.mean() + 10 * self.gradient_penalty(x_hat_out, x_hat)
              loss_d.backward()
              self.opt_d.step()

              #update generator
              self.fake_img = self.G(self.z, alpha)
              self.gen_pred = self.D(self.fake_img, alpha)
              if self.config.loss_type=='lsgan':
                  loss_g = self.gan_loss(self.gen_pred, self.real_label)
              elif self.config.loss_type=='wgan-gp':
                  loss_g = -self.gen_pred.mean()
              loss_g.backward()
              self.opt_g.step()

              if iters%self.config.print_loss_iter==0:
                  print(f"Step: {step} | Iter: {iters} | loss_d: {loss_d.item()}, loss_g: {loss_g.item()}")
              if iters%self.config.save_params_iter==0:
                  self.save_model(iters,step)
              if iters%self.config.save_img_iter==0:
                  self.save_img(iters,step,alpha)
              if iters%self.config.save_log_iter==0:
                  self.writer.add_scalar('loss_g/loss_g', loss_g.item(), iters)
                  self.writer.add_scalar('loss_d/loss_d', loss_d.item(), iters)
            if step!=self.config.max_resl: 
                self.stage_up()
            
            alpha+=1/(self.max_iter//2)
            alpha=min(1,alpha)

    def stage_up(self):
        self.resl+=1
        # self.lr=self.lr*self.config.lr_decay
        self.G.stage_up(self.resl)
        self.D.stage_up(self.resl)
        self.G.to(self.device)
        self.D.to(self.device)

    def gradient_penalty(self, y, x):
        #Compute gradient penalty: (L2_norm(dy/dx) - 1)**2
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                    inputs=x,
                                    grad_outputs=weight,
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def save_model(self, iters, step):
        folder = config.save_root + 'checkpoints'
        file_name = f'ckpt_{2**step}_{iters}'
        path = os.path.join(folder,file_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        ckpt={
          'Generator': self.G.state_dict(),
          'Discriminator': self.D.state_dict(),
          'Optimizer_G': self.opt_g.state_dict(),
          'Optimizer_D': self.opt_d.state_dict(),
        }
        torch.save(ckpt,path)
    
    def save_img(self, iters, step, alpha):
        folder = config.save_root + 'results'
        file_name = f'{2**step}_{iters}.png'
        path = os.path.join(folder,file_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        with torch.no_grad():
            imgs = self.G(self.fixed_z[:self.batchsize],alpha)
            save_image(make_grid((0.5*imgs.cpu()+0.5).clamp(0,1)),path)

    def load(self, train_from):
        checkpoint = torch.load(config.train_from, map_location=self.device)
        self.G.load_state_dict(checkpoint['Generator'],strict=False)
        self.D.load_state_dict(checkpoint['Discriminator'],strict=False)
        self.opt_g.load_state_dict(checkpoint['Optimizer_G'])
        self.opt_d.load_state_dict(checkpoint['Optimizer_D'])

if __name__=='__main__':
    for k, v in vars(config).items():
        print(f'{k}:{v}')
    trainer = Trainer(config)
    if config.train_from:
        #need to modify
        print(f'Loading checkpoint from {config.train_from}')
        trainer.load(config.train_from)
    trainer.train()