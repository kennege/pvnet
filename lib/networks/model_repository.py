from torch import nn
import torch
from torch.nn import functional as F
from lib.networks.resnet import resnet18, resnet50, resnet34

class PVnet(nn.Module):
    def __init__(self, ver_dim, seg_dim, fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32):
        super(PVnet, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet18(fully_conv=True,
                               pretrained=True,
                               output_stride=8,
                               remove_avg_pool_layer=True)

        self.ver_dim=ver_dim
        self.seg_dim=seg_dim

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True)
        )
        self.resnet18_8s = resnet18_8s

        # x8s->128
        self.conv8s=nn.Sequential(
            nn.Conv2d(128+fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up8sto4s=nn.UpsamplingBilinear2d(scale_factor=2)

        # x4s->64
        self.conv4s=nn.Sequential(
            nn.Conv2d(64+s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up4sto2s=nn.UpsamplingBilinear2d(scale_factor=2)

        # x2s->64
        self.conv2s=nn.Sequential(
            nn.Conv2d(64+s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)

        self.convraw = nn.Sequential(
            nn.Conv2d(3+s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(raw_dim, seg_dim+ver_dim, 1, 1)
        )

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x, feature_alignment=False):
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(x)
        fm=self.conv8s(torch.cat([xfc,x8s],1))
        fm=self.up8sto4s(fm)

        fm=self.conv4s(torch.cat([fm,x4s],1))
        fm=self.up4sto2s(fm)

        fm=self.conv2s(torch.cat([fm,x2s],1))
        fm=self.up2storaw(fm)

        x=self.convraw(torch.cat([fm,x],1))
        seg_pred=x[:,:self.seg_dim,:,:]
        ver_pred=x[:,self.seg_dim:,:,:]

        return seg_pred, ver_pred

class ImageEncoder(nn.Module):
    def __init__(self, ver_dim, seg_dim, fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32):
        super(ImageEncoder, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet18(fully_conv=True,
                               pretrained=True,
                               output_stride=8,
                               remove_avg_pool_layer=True)

        self.ver_dim=ver_dim
        self.seg_dim=seg_dim

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True)
        )
        self.resnet18_8s = resnet18_8s

    def forward(self, x, feature_alignment=False):
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(x)

        return x2s, x4s, x8s, xfc

class EstimateEncoder(nn.Module):
    def __init__(self, ver_dim, seg_dim, fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32, pre_trained=True):
        super(EstimateEncoder, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet18(fully_conv=True,
                      pretrained=True,
                      output_stride=8,
                      remove_avg_pool_layer=True)
             
        self.conv1 = nn.Conv2d(18, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.mpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.l1 = nn.Sequential(*list(
            resnet18_8s.children())[4:5])
        self.l2 = nn.Sequential(*list(
            resnet18_8s.children())[5:6])
        self.l3 = nn.Sequential(*list(
            resnet18_8s.children())[6:7])
        self.l4 = nn.Sequential(*list(
            resnet18_8s.children())[7:8])
        self.apool = nn.AvgPool2d(7)
        self.fc = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1,
                               bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )       
        self.ver_dim=ver_dim
        self.seg_dim=seg_dim

    def forward(self, x, feature_alignment=False):

        x = self.conv1(x)   
        x = self.bn1(x)
        
        x2s = self.relu(x)
        x = self.mpool(x2s)
        x4s = self.l1(x)
        x8s = self.l2(x4s)
        x16s = self.l3(x8s)
        x32s = self.l4(x16s)
        x=x32s
      
        xfc = self.fc(x)    

        return x2s, x4s, x8s, xfc

class ImageDecoder(nn.Module):
    def __init__(self, ver_dim, seg_dim, fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32):
        super(ImageDecoder, self).__init__()

        # x8s->128
        self.conv8s=nn.Sequential(
            nn.Conv2d(768, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up8sto4s=nn.UpsamplingBilinear2d(scale_factor=2)
        # x4s->64
        self.conv4s=nn.Sequential(
            nn.Conv2d(256, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up4sto2s=nn.UpsamplingBilinear2d(scale_factor=2)

        # x2s->64
        self.conv2s=nn.Sequential(
            nn.Conv2d(192, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)

        self.convraw = nn.Sequential(
            nn.Conv2d(3+s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(raw_dim, seg_dim+ver_dim, 1, 1)
        )

        self.seg_dim = seg_dim


    def forward(self, x, x2s, x4s, x8s, xfc, x2sEst, x4sEst, x8sEst, xfcEst):

        fm=self.conv8s(torch.cat([xfcEst, xfc, x8s, x8sEst],1))
        fm=self.up8sto4s(fm)

        fm=self.conv4s(torch.cat([fm,x4s, x4sEst],1))
        fm=self.up4sto2s(fm)

        fm=self.conv2s(torch.cat([fm,x2s, x2sEst],1))
        fm=self.up2storaw(fm)

        x=self.convraw(torch.cat([fm,x],1))
        seg_pred=x[:,:self.seg_dim,:,:]
        q_pred=x[:,self.seg_dim:,:,:]

        return seg_pred, q_pred

class EstimateDecoder(nn.Module):
    def __init__(self, ver_dim, seg_dim, fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32):
        super(EstimateDecoder, self).__init__()

        # x8s->128
        self.conv8s=nn.Sequential(
            nn.Conv2d(128+fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up8sto4s=nn.UpsamplingBilinear2d(scale_factor=2)

        # x4s->64
        self.conv4s=nn.Sequential(
            nn.Conv2d(64+s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up4sto2s=nn.UpsamplingBilinear2d(scale_factor=2)

        # x2s->64
        self.conv2s=nn.Sequential(
            nn.Conv2d(64+s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)

        self.convraw = nn.Sequential(
            nn.Conv2d(50, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(raw_dim, ver_dim, 1, 1)
        )

        self.seg_dim = seg_dim

    def forward(self, x, x2s, x4s, x8s, xfc):
        fm=self.conv8s(torch.cat([xfc,x8s],1))
        fm=self.up8sto4s(fm)

        fm=self.conv4s(torch.cat([fm,x4s],1))
        fm=self.up4sto2s(fm)

        fm=self.conv2s(torch.cat([fm,x2s],1))
        fm=self.up2storaw(fm)

        x=self.convraw(torch.cat([fm,x],1))
        ver_pred=x

        return ver_pred, x2s, x4s, x8s

class ImageUNet(nn.Module):
    def __init__(self, ver_dim, seg_dim, fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32):
        super(ImageUNet, self).__init__()
              
        self.imageEncoder = ImageEncoder(ver_dim, seg_dim, fcdim, s8dim, s4dim, s2dim, raw_dim)
        self.imageDecoder = ImageDecoder(ver_dim, seg_dim, fcdim, s8dim, s4dim, s2dim, raw_dim)

    def forward(self, img, x2sEst, x4sEst, x8sEst, xfcEst):
        x2sIm, x4sIm, x8sIm, xfcIm = self.imageEncoder(img)       
        seg_pred, q_pred = self.imageDecoder(img, x2sIm, x4sIm, x8sIm, xfcIm, x2sEst, x4sEst, x8sEst, xfcEst)

        return seg_pred, q_pred
            
class EstimateUNet(nn.Module):
    def __init__(self, ver_dim, seg_dim, fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32):
        super(EstimateUNet, self).__init__()
              
        self.estimateEncoder = EstimateEncoder(ver_dim, seg_dim, fcdim, s8dim, s4dim, s2dim, raw_dim)
        self.estimateDecoder = EstimateDecoder(ver_dim, seg_dim, fcdim, s8dim, s4dim, s2dim, raw_dim)

    def forward(self, vertexEst):
        x2sEst, x4sEst, x8sEst, xfcEst = self.estimateEncoder(vertexEst)
        ver_pred, x2s, x4s, x8s = self.estimateDecoder(vertexEst, x2sEst, x4sEst, x8sEst, xfcEst)

        return ver_pred, x2s, x4s, x8s, xfcEst