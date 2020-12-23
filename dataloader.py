from torch.utils import data
from config import config
from torchvision.datasets import ImageFolder
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
import glob
import os

def data_loader(config,resl):
    imgsize = pow(2,resl)    
    transform=transforms.Compose([transforms.Resize(size=(imgsize,imgsize),interpolation=Image.NEAREST),transforms.ToTensor()])
    batch_list = {4:32, 8:32, 16:32, 32:16, 64:16, 128:16, 256:12, 512:3, 1024:1} # change this according to available gpu memory.
    batchsize = batch_list[pow(2,resl)]
    num_workers = 8
    dataset=FFHQ(config,transform=transform)
    loader=DataLoader(dataset=dataset,batch_size=batchsize,shuffle=False,num_workers=num_workers)
    return loader, batchsize

class FFHQ(data.Dataset):
  def __init__(self, config, transform):
    super(FFHQ,self).__init__()
    self.root = config.root
    self.transform = transform
    self.items = os.listdir(self.root)

  def __len__(self):
      return len(self.items)

  def __getitem__(self, index):
    image = Image.open(os.path.join(self.root, self.items[index]))
    return self.transform(image)