import argparse

parser = argparse.ArgumentParser('PGGAN')

#general settings
parser.add_argument('--root', type=str, default='/home/nas1_userC/yonggyu/dataset/FFHQ/FFHQ_1024/')
parser.add_argument('--save_root', type=str, default='/home/nas1_userE/jihyeonlee/study/pggan/')
parser.add_argument('--random_seed', type=int, default=1211)

#training parameters
parser.add_argument('--lr', type=float, default=0.001)          # learning rate.
parser.add_argument('--lr_decay', type=float, default=0.87)     # learning rate decay at every resolution transition.
parser.add_argument('--nc', type=int, default=3)                # number of input channel.
parser.add_argument('--nz', type=int, default=512)              # input dimension of noise.
parser.add_argument('--ngf', type=int, default=512)             # feature dimension of final layer of generator.
parser.add_argument('--ndf', type=int, default=512)             # feature dimension of first layer of discriminator.
parser.add_argument('--max_resl', type=int, default=10)
parser.add_argument('--max_iter', type=int, default=20000)
parser.add_argument('--print_loss_iter', type=int, default=1000)
parser.add_argument('--save_params_iter', type=int, default=5000)
parser.add_argument('--save_img_iter', type=int, default=5000)

#model structure
parser.add_argument('--flag_bn', type=bool, default=False)          # use of batch-normalization. (not recommended)
parser.add_argument('--flag_pixel', type=bool, default=True)          # use of batch-normalization. (not recommended)
parser.add_argument('--flag_leaky', type=bool, default=True)        # use of leaky relu instead of relu.
parser.add_argument('--flag_tanh', type=bool, default=False)        # use of tanh at the end of the generator.
parser.add_argument('--flag_sigmoid', type=bool, default=False)     # use of sigmoid at the end of the discriminator.

#optimizer settings
parser.add_argument('--optimizer', type=str, default='adam')        # optimizer type.
parser.add_argument('--beta1', type=float, default=0.0)             # beta1 for adam.
parser.add_argument('--beta2', type=float, default=0.99)            # beta2 for adam.

config = parser.parse_args()