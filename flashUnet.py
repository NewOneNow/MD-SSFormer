import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import numpy as np
import cv2
from torchvision.models import resnet50
device = torch.device("cuda:0")
import scipy.stats

class SelfAttention(nn.Module):
    def __init__(self,in_ch, reduce):
        super(SelfAttention, self).__init__()

        self.inner_ch = in_ch // reduce
        self.Q_conv = nn.Conv2d(in_ch, self.inner_ch, kernel_size=1)
        self.K_conv = nn.Conv2d(in_ch, self.inner_ch, kernel_size=1)
        self.V_conv = nn.Conv2d(in_ch, self.inner_ch, kernel_size=1)
        self.restore = nn.Conv2d(self.inner_ch, in_ch, kernel_size=1)

    def forward(self,patch_i):

        b, c, h, w = patch_i.size()
        Q_pro = self.Q_conv(patch_i)  # b c h w
        K_pro = self.K_conv(patch_i)  # b c h w
        V_pro = self.V_conv(patch_i)  # b c h w

        Q_se = Q_pro.view(b, self.inner_ch, -1)  # c hw
        K_se = K_pro.view(b, -1, self.inner_ch)  # hw c
        V_se = V_pro.view(b, self.inner_ch, -1)  # c hw

        weights = F.softmax(torch.matmul(Q_se, K_se), dim=1)  # c c
        att = torch.matmul(weights, V_se)  # c hw
        att = self.restore(att.view(b, -1, h, w)) + patch_i  # b c h w

        return att


class f_test(nn.Module):
    def __init__(self, in_ch):
        super(f_test,self).__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_ch,in_ch//16,1)

    def forward(self,x1,x2):
        x1 = self.conv(x1)
        x2 = self.conv(x2)
        b, c, _, _ = x1.size()
        x1 = x1.detach().cpu().numpy()
        x2 = x2.detach().cpu().numpy()
        counts = 0
        for i in range(b):
            for j in range(c):
                x1_channel = x1[i, j, :, :]
                x2_channel = x2[i, j, :, :]
                x1_channel = x1_channel.flatten()
                x2_channel = x2_channel.flatten()
                F, p_value = scipy.stats.f_oneway(x1_channel,x2_channel)
                alpha = 0.05
                if p_value < alpha:
                    counts+=1
        if counts*2 >= b*c:
            return True
        else:
            return False


class patchViT(nn.Module):
    def __init__(self,in_ch, patchsize, layers):
        super(patchViT, self).__init__()
        self.patchsize = patchsize
        self.layers = layers
        self.att = SelfAttention(in_ch, 4)
        self.f_test = f_test(in_ch)
    def forward(self,x):
        b, c, h, w = x.size()  #
        x1 = x
        numbers1 = h // self.patchsize  # 块数
        for i in range(numbers1):
            for j in range(numbers1):
                patch = x[:, :, i*self.patchsize:(i+1)*self.patchsize, j*self.patchsize:(j+1)*self.patchsize]
                for k in range(self.layers):
                    old = patch
                    if self.f_test(patch, x):
                        patch = old
                        break
                    patch = self.att(patch)
                x1 = x.clone()
                x1[:, :, i*self.patchsize:(i+1)*self.patchsize, j*self.patchsize:(j+1)*self.patchsize] = patch
        return x1


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class erosion_and_dilation(nn.Module):
    def __init__(self):
        super(erosion_and_dilation,self).__init__()
    def erosion(self,img):
        kernel_erosion = np.ones((3, 3), np.uint8)
        eroded_image = cv2.erode(img, kernel_erosion, iterations=1)
        return eroded_image

    def dilated(self, img):
        kernel_dilation = np.ones((3, 3), np.uint8)
        dilated_image = cv2.dilate(img, kernel_dilation, iterations=1)

        return dilated_image

    def forward(self,x):
        b,c,h,w=x.size()
        erision_features = torch.zeros(b, c, h, w)
        dilated_features = torch.zeros(b, c, h, w)
        imgs=[]
        erisions = []
        dilateds = []
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                img=x[i,j,:,:]
                imgs.append(img)
        for i in imgs:
            i=i.detach().cpu().numpy()
            erision = self.erosion(i)
            erisions.append(erision)

            dilated = self.dilated(i)
            dilateds.append(dilated)
        k=0
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                erision_features[i, j, :, :] = torch.from_numpy(erisions[k])
                dilated_features[i, j, :, :] = torch.from_numpy(dilateds[k])
                k=k+1
        return dilated_features,erision_features


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x





class downsample(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(downsample,self).__init__()
        self.conv33=nn.Conv2d(in_ch,out_ch,3,3, padding=1)
        self.conv55=nn.Conv2d(in_ch,out_ch,5,5, padding=1)
        self.ln = nn.LayerNorm(in_ch*2)
        self.trans = self
    def forward(self,x):
        x3 = F.relu(self.conv33(x))
        x5 = F.relu(self.conv55(x))
        xnew = self.ln(torch.cat([x3,x5],dim=1))

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return x * out

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        mid_channel = channel // reduction
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()
        # self.act=SiLU()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        return x * self.sigmoid(avgout + maxout)



class StripedConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(StripedConv,self).__init__()
        self.dilate_erision = erosion_and_dilation()
        #膨胀
        self.Dconv1 = nn.Conv2d(in_ch,out_ch,1)
        self.Dconv31 = nn.Conv2d(in_ch,out_ch,(3,1),padding=(1,0))
        self.Dconv13 = nn.Conv2d(out_ch,out_ch,(1,3),padding=(0,1))
        self.Dconv51 = nn.Conv2d(in_ch,out_ch,(5,1),padding=(2,0))
        self.Dconv15 = nn.Conv2d(out_ch,out_ch,(1,5),padding=(0,2))
        self.Dbn = nn.BatchNorm2d(out_ch*3)
        self.Drelu = nn.ReLU(inplace=True)
        #腐蚀
        self.Econv1 = nn.Conv2d(in_ch, out_ch, 1)
        self.Econv31 = nn.Conv2d(in_ch, out_ch, (3, 1), padding=(1, 0))
        self.Econv13 = nn.Conv2d(out_ch, out_ch, (1, 3), padding=(0, 1))
        self.Econv51 = nn.Conv2d(in_ch, out_ch, (5, 1), padding=(2, 0))
        self.Econv15 = nn.Conv2d(out_ch, out_ch, (1, 5), padding=(0, 2))
        self.Ebn = nn.BatchNorm2d(out_ch*3)
        self.Erelu = nn.ReLU(inplace=True)

        self.depthwise_conv33 = nn.Conv2d(out_ch*3,out_ch*3,3,groups=out_ch*3,stride=1,padding=1)
        self.pointwise_conv11 = nn.Conv2d(out_ch*3,out_ch,1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        dilateX,erisonX = self.dilate_erision(x)
        dilateX = dilateX.cuda()
        erisonX = erisonX.cuda()
        #膨胀
        Dx1=self.Dconv1(dilateX)
        Dx31=self.Dconv31(dilateX)
        Dx3113=self.Dconv13(Dx31)
        Dx51=self.Dconv51(dilateX)
        Dx5115=self.Dconv15(Dx51)
        #腐蚀
        Ex1 = self.Econv1(erisonX)
        Ex31 = self.Econv31(erisonX)
        Ex3113 = self.Econv13(Ex31)
        Ex51 = self.Econv51(erisonX)
        Ex5115 = self.Econv15(Ex51)

        difference = Dx1-Ex51 + Dx1-Ex31 + Dx31-Ex1 + Dx51-Ex1
        dilateX = self.Drelu(self.Dbn(torch.cat([Dx1,Dx3113,Dx5115],dim=1)))
        erisonX = self.Erelu(self.Ebn(torch.cat([Ex1,Ex3113,Ex5115],dim=1)))
        edge = self.relu(self.pointwise_conv11(self.depthwise_conv33(dilateX-erisonX)))

        return edge, difference

class GatedAttention(nn.Module):
    def __init__(self,in_channel):
        super(GatedAttention,self).__init__()
        self.channel = in_channel
        self.inter_channel = in_channel // 32
        self.q_conv = nn.Conv2d(in_channel, self.inter_channel,1)
        self.k_conv = nn.Conv2d(in_channel, self.inter_channel, 1)
        self.v_conv = nn.Conv2d(in_channel, self.inter_channel, 1)
        self.restore = nn.Conv2d(self.inter_channel,in_channel,1)
        self.subnetwork = StripedConv(in_channel,self.inter_channel)

        self.conv = nn.Conv2d(2,1,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ex, dx):
        b, c, h, w = ex.size() #B,C,H,W
        ex = ex.to(device)
        dx = dx.to(device)
        edge, difference = self.subnetwork(dx)  # c//16 h w
        Q = self.q_conv(ex).view(b, self.inter_channel, -1)  # c//16 h*w
        K = self.k_conv(ex).view(b, -1, self.inter_channel)  # h*w c//16

        Wedge = self.sigmoid(self.conv(torch.cat([torch.mean(edge, dim=1, keepdim=True), torch.max(edge, dim=1, keepdim=True).values], dim=1)))

        V = self.v_conv(ex)  # b c h w
        V = V * Wedge + V
        V = V.view(b, self.inter_channel, -1)  # c//16 h*w

        difference = difference.view(b,  -1, self.inter_channel)  # h*w c//16
        weight = torch.matmul(Q, K)  # c//16 c//16
        gate = torch.matmul(Q, difference)  # c//16 c//16

        weights = F.softmax(weight+gate, dim=1).clone()  # c//16 c//16
        atten = torch.matmul(weights, V)  # (c//16,h*w)
        attenX = atten.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        out = self.restore(attenX)+dx
        return out #b,c,h,w

class flashUNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self):
        super(flashUNet, self).__init__()
        in_ch = 3
        out_ch = 2

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = conv_block(in_ch, filters[0])
        #self.SPViT1 = patchViT(filters[0],32,1)

        self.conv2 = conv_block(filters[0], filters[1])
        self.SPViT2 = patchViT(filters[1], 32, 4)

        self.conv3 = conv_block(filters[1], filters[2])
        self.SPViT3 = patchViT(filters[2], 16, 2)

        self.conv4 = conv_block(filters[2], filters[3])
        self.SPViT4 = patchViT(filters[3], 8, 1)


        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])

        self.att5 = GatedAttention(filters[3])

        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])

        self.att4 = GatedAttention(filters[2])

        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])

        self.att3 = GatedAttention(filters[1])

        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.conv1(x)
        #e1 = self.SPViT1(e1)

        e2 = self.Maxpool1(e1)
        e2 = self.conv2(e2)
        e2 = self.SPViT2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.conv3(e3)
        e3 = self.SPViT3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.conv4(e4)
        e4 = self.SPViT4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = self.att5(e4, d5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = self.att4(e3, d4)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.att3(e2, d3)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out