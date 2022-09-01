import torch
import torch.nn as nn
import torch.nn.functional as F

# imports for padding module(s)
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

def round_grad(x):
    # TODO: cite something? (this function is based off of Thor's code)

    x_hard = torch.round(x)
    x = (x_hard - x.data) + x

    return x


class Pad_Conv2d(nn.Module):
    __doc__ = """Pytorch implementation of Keras' 'same' padding for Conv2d layers. Add this just before Conv2d layers to apply the padding."""

    def __init__(self, kernel_size: _size_2_t, stride: _size_2_t):
        super().__init__()

        self.k = _pair(kernel_size)  # (H,W)
        self.s = _pair(stride)  # (H,W)

    def forward(self, x):
        dims = x[0,0,:,:].shape

        # Find total padding along each axis
        if dims[1] % self.s[0] == 0:
            pad_v = max(self.k[0] - self.s[0], 0)
        else:
            pad_v = max(self.k[0] - (dims[1] % self.s[0]), 0)

        if dims[0] % self.s[1] == 0:
            pad_h = max(self.k[1] - self.s[1], 0)
        else:
            pad_h = max(self.k[1] - (dims[0] % self.s[1]), 0)

        # Find padding on each side
        pad_top = pad_v // 2
        pad_bot = pad_v - pad_top
        pad_left = pad_h // 2
        pad_right = pad_h - pad_left

        return F.pad(x, (pad_left, pad_right, pad_top, pad_bot))


class Pad_ConvTranspose2d(nn.Module):
    __doc__ = """Pytorch implementation of Keras' 'same' padding for Conv2d layers. Add this just before Conv2d layers to apply the padding."""

    def __init__(self, kernel_size: _size_2_t, stride: _size_2_t):
        super().__init__()

        self.k = _pair(kernel_size)  # (H,W)
        self.s = _pair(stride)  # (H,W)

    def forward(self, x):
        dims = x[0,0,:,:].shape

        # Find total padding along each axis
        if dims[1] % self.s[0] == 0:
            pad_v = max(self.k[0] - self.s[0], 0)
        else:
            pad_v = max(self.k[0] - (dims[1] % self.s[0]), 0)

        if dims[0] % self.s[1] == 0:
            pad_h = max(self.k[1] - self.s[1], 0)
        else:
            pad_h = max(self.k[1] - (dims[0] % self.s[1]), 0)

        # Find padding on each side
        pad_top = pad_v // 2
        pad_bot = pad_v - pad_top
        pad_left = pad_h // 2
        pad_right = pad_h - pad_left

        return F.pad(x, (pad_left, pad_right, pad_top, pad_bot))


class Critic(nn.Module):
    def __init__(self, f, k, s, p, device='cpu'):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            self.block(f[0], f[1], 4, 2, norm=False, device=device),
            self.block(f[1], f[2], 4, 2, device=device),
            self.block(f[2], f[3], 4, 2, device=device),
            self.block(f[3], f[4], 4, 2, device=device),
            self.block(f[4], f[5], 4, 1, device=device),
            Pad_Conv2d(4, 1),
            nn.Conv2d(f[5], f[6], 4, 1, device=device),
        )
        

    def block(self, f_in, f_out, k, s=1, norm=True, device='cpu'):

        block = nn.Sequential()
        block.append(Pad_Conv2d(k, s))
        block.append(nn.Conv2d(f_in, f_out, k, s, device=device))
        if norm:
            block.append(nn.BatchNorm2d(f_out, eps=0.001, momentum=0.99, device=device))
        block.append(nn.LeakyReLU(0.2))

        return block

    def _rescale_endpoints(self, curr):
        scale = (256.0/curr.shape[-2], 256.0/curr.shape[-1])

        ends = torch.zeros((curr.shape[0],1,256,256))
        for i in range(curr.shape[0]):
            # Find endpoints
            coords = torch.argwhere(curr[i,0,:,:] == 2, )

            # Translate endpoints
            coords[:,0] = coords[:,0] * scale[0]
            coords[:,1] = coords[:,1] * scale[1]

            # Connect endpoints with ones
            x1 = coords[0,1].item()
            x2 = coords[1,1].item()
            y1 = coords[0,0].item()
            y2 = coords[1,0].item()

            x = x1
            y = y1

            if (x1 < x2):
                x_dir = 1
            else:
                x_dir = -1

            if (y1 < y2):
                y_dir = 1
            else:
                y_dir = -1

            # Determine y from x
            if x2-x1 != 0:
                m = (y2-y1)/(x2-x1)
                while x != x2:
                    y = round_grad(m*(x-x1) + y1)
                    ends[i,0,y,x] = 1
                    x += x_dir
            else:
                while x != x2:
                    ends[i,0,y1,x] = 1
                    x += x_dir


            x = x1
            y = y1

            # Determine x from y
            if y2-y1 != 0:
                m = (x2-x1)/(y2-y1)
                while y != y2:
                    x = round_grad(m*(y-y1) + x1)
                    ends[i,0,y,x] = 1
                    y += y_dir
            else:
                while y != y2:
                    ends[i,0,y,x1] = 1
                    y += y_dir

            
            ends[i,0,coords[0,0],coords[0,1]] = 2
            ends[i,0,coords[1,0],coords[1,1]] = 2
        return ends


    def forward(self, x):

        ends = self._rescale_endpoints(x[:,1:2,:,:])    # Rescale endpoints matrix while preserving 1-width path and endpoints
        x = F.interpolate(x, (x.shape[0],x.shape[1],256,256))   # Rescale input matrix
        x[:,1:2,:,:] = ends     # replace current endpoints channel with the properly rescaled one

        y = self.net(x)
        return y



class Generator(nn.Module):
    def __init__(self, f, k, s, p, device='cpu'):
        super(Generator, self).__init__()

        # Encoder
        self.e1 = self.enc(f[0], f[1], 4, 2, norm=False, device=device)
        self.e2 = self.enc(f[1], f[2], 4, 2, device=device)
        self.e3 = self.enc(f[2], f[3], 4, 2, device=device)
        self.e4 = self.enc(f[3], f[4], 4, 2, device=device)
        self.e5 = self.enc(f[4], f[5], 4, 2, device=device)
        self.e6 = self.enc(f[5], f[6], 4, 2, device=device)
        self.e7 = self.enc(f[6], f[7], 4, 2, device=device)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            Pad_Conv2d(4,2),
            nn.Conv2d(f[7], f[8], 4, 2, device=device),
            nn.ReLU()
        )

        # Decoder
        self.dec_act = nn.ReLU()
        self.d1 = self.dec(f[8], f[9], 4, 2, dropout=True, device=device)
        self.d2 = self.dec(f[9]+f[7], f[10], 4, 2, dropout=True, device=device)
        self.d3 = self.dec(f[10]+f[6], f[11], 4, 2, dropout=True, device=device)
        self.d4 = self.dec(f[11]+f[5], f[12], 4, 2, device=device)
        self.d5 = self.dec(f[12]+f[4], f[13], 4, 2, device=device)
        self.d6 = self.dec(f[13]+f[3], f[14], 4, 2, device=device)
        self.d7 = self.dec(f[14]+f[2], f[15], 4, 2, device=device)

        # Final output layer
        self.out = nn.Sequential(
            nn.ConvTranspose2d(f[15]+f[1], f[16], 4, 2, 1, device=device),
            nn.Tanh()
        )

    
    def enc(self, f_in, f_out, k, s=1, norm=True, device='cpu'):

        block = nn.Sequential()
        block.append(Pad_Conv2d(k,s))
        block.append(nn.Conv2d(f_in, f_out, k, s, device=device))
        if norm:
            block.append(nn.BatchNorm2d(f_out, eps=0.001, momentum=0.99, device=device))
        block.append(nn.LeakyReLU(0.2))

        return block
    

    # Each pass through a decoder block should be followed by concatenation of skip tensor then leaky ReLU
    def dec(src, f_in, f_out, k, s=1, p=1, dropout=False, device='cpu'):
        
        block = nn.Sequential()
        # block.append(Pad_Conv2d(k,s))
        block.append(nn.ConvTranspose2d(f_in, f_out, k, s, p, device=device))
        block.append(nn.BatchNorm2d(f_out, eps=0.001, momentum=0.99, device=device))
        if dropout:
            block.append(nn.Dropout(0.4))

        return block
        
    def _rescale_endpoints(self, curr):
        scale = (256.0/curr.shape[-2], 256.0/curr.shape[-1])

        ends = torch.zeros((curr.shape[0],1,256,256))
        for i in range(curr.shape[0]):
            # Find endpoints
            coords = torch.argwhere(curr[i,0,:,:] == 2)

            # Translate endpoints
            coords[:,0] = coords[:,0] * scale[0]
            coords[:,1] = coords[:,1] * scale[1]
            # coords = torch.round(coords)

            # Connect endpoints with ones
            x1 = coords[0,1].item()
            x2 = coords[1,1].item()
            y1 = coords[0,0].item()
            y2 = coords[1,0].item()

            x = x1
            y = y1

            if (x1 < x2):
                x_dir = 1
            else:
                x_dir = -1

            if (y1 < y2):
                y_dir = 1
            else:
                y_dir = -1

            # Determine y from x
            if x2-x1 != 0:
                m = (y2-y1)/(x2-x1)
                while x != x2:
                    y = round_grad(m*(x-x1) + y1)
                    ends[i,0,y,x] = 1
                    x += x_dir
            else:
                while x != x2:
                    ends[i,0,y1,x] = 1
                    x += x_dir


            x = x1
            y = y1

            # Determine x from y
            if y2-y1 != 0:
                m = (x2-x1)/(y2-y1)
                while y != y2:
                    x = round_grad(m*(y-y1) + x1)
                    ends[i,0,y,x] = 1
                    y += y_dir
            else:
                while y != y2:
                    ends[i,0,y,x1] = 1
                    y += y_dir

            
            ends[i,0,coords[0,0],coords[0,1]] = 2
            ends[i,0,coords[1,0],coords[1,1]] = 2
        return ends





    def forward(self, x):
        ends = self._rescale_endpoints(x[:,:1,:,:])    # Rescale endpoints matrix while preserving 1-width path and endpoints
        x = F.interpolate(x, (x.shape[0],x.shape[1],256,256))   # Rescale input matrix
        x[:,:1,:,:] = ends     # replace current endpoints channel with the properly rescaled one

        # encoder
        s1 = self.e1(x)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        s5 = self.e5(s4)
        s6 = self.e6(s5)
        s7 = self.e7(s6)

        # Bottleneck
        y = self.bottleneck(s7)


        # Decoder
        y = self.d1(y)
        y = torch.concat((y, s7), axis=1)
        y = self.dec_act(y)

        y = self.d2(y)
        y = torch.concat((y, s6), axis=1)
        y = self.dec_act(y)

        y = self.d3(y)
        y = torch.concat((y, s5), axis=1)
        y = self.dec_act(y)

        y = self.d4(y)
        y = torch.concat((y, s4), axis=1)
        y = self.dec_act(y)

        y = self.d5(y)
        y = torch.concat((y, s3), axis=1)
        y = self.dec_act(y)

        y = self.d6(y)
        y = torch.concat((y, s2), axis=1)
        y = self.dec_act(y)

        y = self.d7(y)
        y = torch.concat((y, s1), axis=1)
        y = self.dec_act(y)

        y = self.out(y)

        y = self._round(y)
        return y
    

    def _round(self, mat):
        # TODO: cite something? (this function is based off of Thor's code)

        paths = mat[:,:1,:,:]
        maps = mat[:,-1:,:,:]

        # Confine path to [0,1] then round
        paths = paths + 1
        paths = paths / 2
        paths = round_grad(paths)

        mat = torch.concat((paths, maps), axis=1)
        return mat