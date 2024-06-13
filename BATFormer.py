import numpy as np
np.set_printoptions(threshold=1000)

from torchvision.ops import roi_align, nms

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
np.set_printoptions(threshold=1000)

""" Parts of the U-Net model """
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)

class SingleDepthConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)


class DoubleDepthConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Downs(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            SingleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Ups(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = SingleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up3(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, x3):
        x1 = self.up4(x1)
        x2 = self.up2(x2)
        # input is BCHW
        diffY1 = x3.size()[2] - x1.size()[2]
        diffX1 = x3.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX1 // 2, diffX1 - diffX1 // 2,
                        diffY1 // 2, diffY1 - diffY1 // 2])
        # ----------------------------------------------------
        diffY2 = x3.size()[2] - x2.size()[2]
        diffX2 = x3.size()[3] - x2.size()[3]
        x2 = F.pad(x2, [diffX2 // 2, diffX2 - diffX2 // 2,
                        diffY2 // 2, diffY2 - diffY2 // 2])

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x3, x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SkipConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels,  kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        return out1+out2+out3

#  ================================================== for Res Unet ===================================================
# reference: https://github.com/rishikksh20/ResUnet

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)

#  ================================================== for Attention Unet ===================================================
# reference: https://github.com/LeeJunHyun/Image_Segmentation

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        #self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.fn(x, **kwargs)
        return self.norm(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            #nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

#  ====================================================================================================================

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

#  ===================================================================================================================

class TransformerDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, image_size, depth=2, dmodel=1024, mlp_dim=2048, patch_size=2, heads=6, dim_head=128, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel*4

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, self.dmodel),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout, num_patches)

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height//patch_height),
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        # transformer layer
        ax = self.transformer(x)
        out = self.recover_patch_embedding(ax)
        return out
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class FixCNN(nn.Module):
    def __init__(self, win_size=16):
        super(FixCNN, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, 1, win_size, win_size))
        #self.bias = nn.Parameter(torch.zeros(0))
    def forward(self, x):
        out = F.conv2d(x, self.weight, bias=None, stride=1, padding=0)
        return out

def Shifted_Windows(height, width, win_size, stride=2):
    shift_y = torch.arange(0, height-win_size+1, stride)
    shift_x = torch.arange(0, width-win_size+1, stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()), axis=1)
    M = shift.shape[0]
    window = shift.reshape(M, 4)
    window[:, 2] = window[:, 0] + win_size-1
    window[:, 3] = window[:, 1] + win_size-1
    return window

def make_gridsx(win_size):
    shift_y = torch.arange(0, win_size, 1)
    shift_x = torch.arange(0, win_size, 1)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    return torch.tensor(shift_x)

def make_gridsy(win_size):
    shift_y = torch.arange(0, win_size, 1)
    shift_x = torch.arange(0, win_size, 1)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    return torch.tensor(shift_y)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNorm3p(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm5 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x5, x4, x3, **kwargs):
        return self.fn(self.norm5(x5), self.norm4(x4), self.norm3(x3), **kwargs)

class PreNorm2p(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x1, x2, **kwargs):
        return self.fn(self.norm1(x1), self.norm2(x2), **kwargs)

class PreNorm2pm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, mask, **kwargs):
        return self.fn(self.norm(x), mask, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Attention_group(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., r=2):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.r = r

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim*self.r, inner_dim, bias=False)
        self.to_v = nn.Linear(dim*self.r, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=self.heads)
        x_kv = rearrange(x, 'b (n r) d -> b n (d r)', r=self.r)
        k = rearrange(self.to_k(x_kv), 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(self.to_v(x_kv), 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Attention_global(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q1 = nn.Linear(dim, inner_dim, bias=False)
        self.to_k1 = nn.Linear(dim, inner_dim, bias=False)
        self.to_v1 = nn.Linear(dim, inner_dim, bias=False)
        self.to_q2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_k2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_v2 = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim*2, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x5, x4, x3):
        q1 = self.to_q1(x3)
        q1 = rearrange(q1, 'b n (h d) -> b h n d', h=self.heads)
        q2 = self.to_q2(x3)
        q2 = rearrange(q2, 'b n (h d) -> b h n d', h=self.heads)

        k1 = self.to_k1(x4)
        k1 = rearrange(k1, 'b n (h d) -> b h n d', h=self.heads)
        v1 = self.to_v1(x4)
        v1 = rearrange(v1, 'b n (h d) -> b h n d', h=self.heads)

        k2 = self.to_k1(x5)
        k2 = rearrange(k2, 'b n (h d) -> b h n d', h=self.heads)
        v2 = self.to_v1(x5)
        v2 = rearrange(v2, 'b n (h d) -> b h n d', h=self.heads)

        dots1 = torch.matmul(q1, k1.transpose(-1, -2)) * self.scale
        attn1 = self.attend(dots1)
        out1 = torch.matmul(attn1, v1)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')

        dots2 = torch.matmul(q2, k2.transpose(-1, -2)) * self.scale
        attn2 = self.attend(dots2)
        out2 = torch.matmul(attn2, v2)
        out2 = rearrange(out2, 'b h n d -> b n (h d)')

        out = torch.cat((out1, out2), dim=-1)

        return self.to_out(out)

class Attention_local(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., win_size=16, img_height=256, img_width=256):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.win_size = win_size

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        # --------------------- new sub-networks ---------------------
        self.fixcnn = FixCNN(win_size=win_size//2)
        self.window = Shifted_Windows(img_height, img_width, win_size)
        self.shifty = make_gridsy(win_size).cuda()
        self.shiftx = make_gridsx(win_size).cuda()

    def forward(self, x, prob):
        b, c, h, w = prob.shape
        _, N, d = x.shape
        log_prob = torch.log2(prob + 1e-10)
        entropy = -1 * torch.sum(prob * log_prob, dim=1)  # b h w
        # x (b n d) -> x (b d h w) for making window easily
        x_2d = rearrange(x, 'b (h w) d -> b d h w', h=2 * h, w=2 * w)
        outx_2d = x_2d * 0  # b d h w
        win_cunt = x_2d[:, 0, :, :] * 0 # b h w

        # compute the score of each window, achieve by fix the filters
        win_score = self.fixcnn(entropy[:, None, :, :])/(self.win_size//2*self.win_size//2) # b 1 h w
        win_score = win_score.view(b, -1)
        window = torch.from_numpy(self.window).cuda().float() # N 4
        keep_num = min(int(0.7*(2 * h // self.win_size)**2), 50) #120
        for i in range(b):
            scorei = win_score[i]  # N
            indexi = nms(boxes=window, scores=scorei, iou_threshold=0.2)
            indexi = indexi[:keep_num]
            keep_windowi = window[indexi, :]

            #visual_box(keep_windowi, "layer2", 2)

            window_batch_indexi = torch.zeros(keep_windowi.shape[0]) + i
            window_batch_indexi = window_batch_indexi.cuda().float()
            index_windowi = torch.cat([window_batch_indexi[:, None], keep_windowi], dim=1)
            window_featurei = roi_align(x_2d, index_windowi, (self.win_size, self.win_size))  # b d h w
            xi = rearrange(window_featurei, 'm d h w -> m (h w) d')

            qkvi = self.to_qkv(xi).chunk(3, dim=-1)
            qi, ki, vi = map(lambda t: rearrange(t, 'm n (h d) -> m h n d', h=self.heads), qkvi)
            dotsi = torch.matmul(qi, ki.transpose(-1, -2)) * self.scale
            attni = self.attend(dotsi)
            outi = torch.matmul(attni, vi)
            outi = rearrange(outi, 'm h n d -> m n (h d)')
            outi = self.to_out(outi)  # m n d
            # only those window area can add out on original x, out_x_2d [b d h w]
            outi_2d = rearrange(outi, 'm (h w) d -> m d h w', h=self.win_size)

            m = outi.shape[0]
            for j in range(m):
                sy = int(keep_windowi[j, 1])
                sx = int(keep_windowi[j, 0])
                outx_2d[i, :, sy:sy+self.win_size, sx:sx+self.win_size] += outi_2d[j, :, :, :]
                win_cunt[i, sy:sy+self.win_size, sx:sx+self.win_size] += 1
        outx = rearrange(outx_2d/(win_cunt[:, None, :, :] + 1e-10), 'b d h w -> b (h w) d')
        x = x + outx
        return x

class Transformer_naive(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention_group(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Transformer_global(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm3p(dim, Attention_global(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x5, x4, x):
        for attn, ff in self.layers:
            x = attn(x5, x4, x) + x
            x = ff(x) + x
        return x

class Transformer_local(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128, win_size=16, img_height=256, img_width=256):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm2pm(dim, Attention_local(dim, heads=heads, dim_head=dim_head, dropout=dropout, win_size=win_size, img_height=img_height, img_width=img_width)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x, fore_score):
        for attn, ff in self.layers:
            x = attn(x, fore_score)  # + x
            x = ff(x) + x
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class Transformer_block_global(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, image_size, depth=2, dmodel=1024, mlp_dim=2048, patch_size=2, heads=6, dim_head=128, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel*4

        self.to_patch_embedding_x5 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),
            nn.Linear(in_channels*4, self.dmodel),
        )
        self.to_patch_embedding_x4 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=2, p2=2),
            nn.Linear(in_channels*2*4, self.dmodel), # 2 means the channel of x4 is double of channel of x3, 4 is p1*p2
        )
        self.to_patch_embedding_x3 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, self.dmodel),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer_global(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout, num_patches)

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height//patch_height),
        )

    def forward(self, x5, x4, x3):
        x5 = self.to_patch_embedding_x5(x5)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        x4 = self.to_patch_embedding_x4(x4)
        x3 = self.to_patch_embedding_x3(x3)

        _, n5, _ = x5.shape
        _, n4, _ = x4.shape
        _, n3, _ = x3.shape

        x5 += self.pos_embedding[:, :n5]
        x4 += self.pos_embedding[:, :n4]
        x3 += self.pos_embedding[:, :n3]

        x5 = self.dropout(x5)
        x4 = self.dropout(x4)
        x3 = self.dropout(x3)
        # transformer layer
        ax = self.transformer(x5, x4, x3)
        out = self.recover_patch_embedding(ax)
        return out


class Transformer_block_local(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, image_size, depth=2, n_classes=9, patch_size=1, win_size=16, heads=6, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel*4

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, self.dmodel),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        #self.softmax = nn.Softmax(dim=1)

        self.transformer = Transformer_local(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout, num_patches, win_size, image_height, image_width)

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height//patch_height),
        )

    def forward(self, x1, x2):
        x = self.to_patch_embedding(x1)  # (b, c, h, w) -> (b, num_patches, dim_patches)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)

        # make the guided range of self attention
        #x2 = self.softmax(x2)  # (b, c, h, w)

        # transformer layer
        ax = self.transformer(x, x2)
        out = self.recover_patch_embedding(ax)
        return out

class Transformer_block(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, image_size, depth=2, dmodel=1024, mlp_dim=2048, patch_size=2, heads=6, dim_head=128, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel*4

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, self.dmodel),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout, num_patches)

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height//patch_height),
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        # transformer layer
        ax = self.transformer(x)
        out = self.recover_patch_embedding(ax)
        return out
class C2FTrans(nn.Module):
    def __init__(self, global_block, local_block, layers, n_channels, n_classes, imgsize, patch_size=2, bilinear=True):
        super(C2FTrans, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.scale = 4  # 1 2 4

        self.inc = DoubleConv(n_channels, 64 // self.scale)
        self.down1 = Down(64 // self.scale, 128 // self.scale)
        self.down2 = Down(128 // self.scale, 256 // self.scale)
        self.down3 = Down(256 // self.scale, 512 // self.scale)
        factor = 2 if bilinear else 1
        self.down4 = Down(512 // self.scale, 1024 // factor // self.scale)

        self.up4 = Up(1024 // self.scale, 512 // factor // self.scale, bilinear)
        self.up3 = Up(512 // self.scale, 256 // factor // self.scale, bilinear)
        self.up2 = Up(256 // self.scale, 128 // factor // self.scale, bilinear)
        self.up1 = Up(128 // self.scale, 64 // self.scale, bilinear)

        self.softmax = nn.Softmax(dim=1)

        for p in self.parameters():
            p.requires_grad = True # set "True" manually in the first 350 epochs, then load the best model and set "False" manually in the following 50 epochs.

        self.trans_local2 = local_block(128 // self.scale // factor, 128 // self.scale // factor * 2, imgsize // 2, 1, heads=6, patch_size=1, n_classes=n_classes, win_size=16)
        self.trans_global = global_block(256 // factor // self.scale, 256 // factor // self.scale * 2, imgsize // 4, 1, heads=4, patch_size=1)

        self.outc1 = OutConv(64 // self.scale * 4, n_classes)
        self.convl1 = nn.Conv2d(64 // self.scale, 64 // self.scale * 4, kernel_size=1, padding=0, bias=False)
        self.outc2 = OutConv(64 // self.scale * 4, n_classes)

        self.convl2 = nn.Conv2d(128//factor//self.scale*2, 64//self.scale*4, kernel_size=1, padding=0, bias=False)
        #self.convl2 = nn.Conv2d(128 // factor // self.scale, 64 // self.scale * 4, kernel_size=1, padding=0, bias=False)

        self.outc3 = OutConv(64 // self.scale * 4, n_classes)
        self.convl3 = nn.Conv2d(256//factor//self.scale*2, 64//self.scale*4, kernel_size=1, padding=0, bias=False)
        #self.convl3 = nn.Conv2d(256 // factor // self.scale, 64 // self.scale * 4, kernel_size=1, padding=0, bias=False)
        self.out = OutConv(3 * 64 // self.scale * 4, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d4 = self.up4(x5, x4)
        d3 = self.up3(d4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)

        trans_global = self.trans_global(x5, d4, d3)
        l3 = self.convl3(trans_global)
        #l3 = self.convl3(d3)

        pred3 = self.outc3(l3)  # b c h w
        l3_up = l3[:, :, :, :, None].repeat(1, 1, 1, 1, 16)
        l3_up = rearrange(l3_up, 'b c h w (m n) -> b c (h m) (w n)', m=4, n=4)
        #pred3_up = pred3[:, :, :, :, None].repeat(1, 1, 1, 1, 16)
        #pred3_up = rearrange(pred3_up, 'b c h w (m n) -> b c (h m) (w n)', m=4, n=4)

        pred3_p = self.softmax(pred3)
        trans_local2 = self.trans_local2(d2, pred3_p)
        l2 = self.convl2(trans_local2)
        #l2 = self.convl2(d2)

        pred2 = self.outc2(l2)  # b c h w
        l2_up = l2[:, :, :, :, None].repeat(1, 1, 1, 1, 4)
        l2_up = rearrange(l2_up, 'b c h w (m n) -> b c (h m) (w n)', m=2, n=2)
        #pred2_up = pred2[:, :, :, :, None].repeat(1, 1, 1, 1, 4)
        #pred2_up = rearrange(pred2_up, 'b c h w (m n) -> b c (h m) (w n)', m=2, n=2)

        l1 = self.convl1(d1)
        pred1 = self.outc1(l1)  # b c h w

        predf = torch.cat((l1, l2_up, l3_up), dim=1)
        predf = self.out(predf)

        return predf

def C2FTransformer(pretrained=False):
    model = C2FTrans(Transformer_block_global, Transformer_block_local, [1, 1, 1, 1], 1,2,256)
    return model

c2 = C2FTransformer().cuda()
x= torch.zeros(2,1,256,256).cuda()
print(c2(x).shape)