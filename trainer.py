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

class Trainer:
  def __init__(self, config):
    self.resl = 2
    self.epoch = 0
    self.config=config
    self.lr=self.config.lr
    self.max_iter=self.config.max_iter
    self.G = model.Generator(config)
    self.D = model.Discriminator(config)
    print('Generator structure: ')
    print(self.G.main_layers, self.G.to_rgb_layers)
    print('Discriminator structure: ')
    print(self.D.from_rgb_layers, self.D.main_layers)
    #wgan-gp
    #exponential moving average??
    #spectral norm?
    self.mse = nn.MSELoss()
    self.device = torch.device("cuda")
    betas=(self.config.beta1, self.config.beta2)
    self.opt_g=Adam(self.G.parameters(), lr=self.lr, betas=betas, weight_decay=0.0)
    self.opt_d=Adam(self.D.parameters(), lr=self.lr, betas=betas, weight_decay=0.0)
    self.fixed_z = torch.rand(64,self.config.nz,1,1).to(self.device)

  def train(self):
    for step in range(2, self.config.max_resl+1):
      loader, self.batchsize = DL.data_loader(self.config,self.resl)
      loader=iter(loader)
      self.real_label = torch.ones(self.batchsize).to(self.device)
      self.fake_label = torch.zeros(self.batchsize).to(self.device)
      alpha=0.0
      if step==2: alpha=1
      for iters in tqdm(range(0,self.max_iter)):
        self.G.to(self.device)
        self.D.to(self.device)
        self.G.zero_grad()
        self.D.zero_grad()
        self.real_img=next(loader)
        self.real_img = self.real_img.to(self.device)
        self.z = torch.rand(self.batchsize, self.config.nz).to(self.device)

        #update discriminator
        self.fake_img = self.G(self.z, alpha)
        self.real_pred = self.D(self.real_img, alpha)
        self.fake_pred = self.D(self.fake_img.detach(), alpha)
        loss_d = self.mse(self.real_pred, self.real_label)+self.mse(self.fake_pred,self.fake_label)
        loss_d.backward()
        self.opt_d.step()

        #update generator
        self.gen_pred = self.D(self.fake_img, alpha)
        loss_g = self.mse(self.gen_pred, self.real_label)
        loss_g.backward()
        self.opt_g.step()

        if iters%self.config.print_loss_iter==0:
          print(f"Step: {step} | Iter: {iters} | loss_d: {loss_d.item()}, loss_g: {loss_g.item()}")
        if iters%self.config.save_params_iter==0:
          self.save_model(iters,step)
        if iters%self.config.save_img_iter==0:
          self.save_img(iters,step,alpha)

      if step!=self.config.max_resl: 
        self.stage_up()
      
      alpha+=1/(self.max_iter//2)
      alpah=min(1,alpha)

  def stage_up(self):
    self.resl+=1
    # self.lr=self.lr*self.config.lr_decay
    self.G.stage_up(self.resl)
    self.D.stage_up(self.resl)
    self.G.to(self.device)
    self.D.to(self.device)

  def save_model(self, iters, step):
    #restart todo
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
      save_image(make_grid(0.5*imgs.cpu()+0.5),path)

  def save_tb(self):
    #tensorboard todo
    pass


if __name__=='__main__':
  for k, v in vars(config).items():
    print(f'{k}:{v}')
  trainer = Trainer(config)
  trainer.train()
