import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from generator import Generator
from IPython.display import Image

parser = argparse.ArgumentParser()
parser.add_argument('--netG', default='', help="path to the generator network")
parser.add_argument('--batchSize', type=int, default=64, help='number of samples to produce')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outImg', default='fake_sample.png', help='name of the generated fake image')
parser.add_argument('--display', default='1', help='enables display of the fake generated image')

opt = parser.parse_args()
print(opt)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3

netG = Generator(ngpu=opt.ngpu, nz=opt.nz, ngf=opt.ngf, nc=opt.nc)
netG.load_state_dict(torch.load(opt.netG))
netG.cuda()
netG.eval()

fixed_noise = torch.randn(batchSize, nz, 1, 1, device=device)
fake = netG(fixed_noise)
vutils.save_image(fake.detach(), '%s' % (opt.outImg), normalize=True)
if opt.display:
  Image(opt.outImg)
