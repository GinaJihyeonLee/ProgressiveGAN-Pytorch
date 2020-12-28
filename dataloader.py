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
    batch_list = {4:128, 8:128, 16:128, 32:128, 64:64, 128:32, 256:16, 512:8, 1024:4} # change this according to available gpu memory.
    batchsize = batch_list[pow(2,resl)]
    num_workers = 32
    dataset=FFHQ(config,transform=transform)
    loader=DataLoader(dataset=dataset,batch_size=batchsize,shuffle=True,num_workers=num_workers, drop_last=True)
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