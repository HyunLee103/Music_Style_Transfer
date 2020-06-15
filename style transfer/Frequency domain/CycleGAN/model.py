import torch.nn as nn
import torch.nn.functional as F
from layer import *

class Generator(nn.Module):
    def __init__(self, in_channels,out_channels,nker=64,norm='inorm'):
        super(Generator, self).__init__()

        # encoder
        # Leaky relu 사용, 첫번째 encoder는 batchnorm X
        self.enc1 = CBR2d(in_channels,1*nker,kernel_size=4, padding=1,stride=2,
        norm = None,relu=0.2)
        self.enc2 = CBR2d(1*nker,2*nker,kernel_size=4, padding=1,stride=2,
        norm = norm ,relu=0.2)
        self.enc3 = CBR2d(2*nker,4*nker,kernel_size=4, padding=1,stride=2,
        norm = norm,relu=0.2)
        self.enc4 = CBR2d(4*nker,8*nker,kernel_size=4, padding=1,stride=2,
        norm = norm,relu=0.2)
        self.enc5 = CBR2d(8*nker,8*nker,kernel_size=4, padding=1,stride=2,
        norm = norm,relu=0.2)
        self.enc6 = CBR2d(8*nker,8*nker,kernel_size=4, padding=1,stride=2,
        norm = norm,relu=0.2)


        # decoder, skip-connection 고려해서 input channel modeling
        self.dec1 = DECBR2d(8*nker, 8*nker, kernel_size=4, padding=1,
        norm = norm, relu=0.0, stride=2)
        self.drop1 = nn.Dropout2d(0.5)
        self.pad1 = nn.ReflectionPad2d((0,1,0,0)) # (left, right, top, bottom) 
        
        self.dec2 = DECBR2d(2 * 8 * nker, 8*nker, kernel_size=4, padding=1,
        norm = norm, relu=0.0, stride=2)
        self.drop2 = nn.Dropout2d(0.5)
        self.pad2 = nn.ReflectionPad2d((0,0,0,1))

        self.dec3 = DECBR2d(2*8*nker, 4*nker, kernel_size=4, padding=1,
        norm = norm, relu=0.0, stride=2)
        self.drop3 = nn.Dropout2d(0.5)
        self.pad3 = nn.ReflectionPad2d((1,0,0,0))

        self.dec4 = DECBR2d(2 * 4 *nker, 2*nker, kernel_size=4, padding=1,
        norm = norm, relu=0.0, stride=2)
        self.pad4 = nn.ReflectionPad2d((0,0,1,0))

        self.dec5 = DECBR2d(2*2*nker, 1*nker, kernel_size=4, padding=1,
        norm = norm, relu=0.0, stride=2)
        self.pad5 = nn.ReflectionPad2d((1,0,0,0))
        
        self.dec6 = DECBR2d(2*1*nker, out_channels, kernel_size=4, padding=1,
        norm = None, relu=None, stride=2)

    def forward(self, x):
        
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
     

        dec1 = self.dec1(enc6)
        drop1 = self.drop1(dec1)
        pad1 = self.pad1(drop1)

        cat2 = torch.cat((pad1,enc5),dim=1)
        dec2 = self.dec2(cat2)
        drop2 = self.drop2(dec2)
        pad2 = self.pad2(drop2)

        cat3 = torch.cat((pad2,enc4),dim=1)
        dec3 = self.dec3(cat3)
        drop3 = self.drop3(dec3)
        pad3 = self.pad3(drop3)

        cat4 = torch.cat((pad3,enc3),dim=1)
        dec4 = self.dec4(cat4)
        pad4 = self.pad4(dec4)

        cat5 = torch.cat((pad4,enc2),dim=1)
        dec5 = self.dec5(cat5)
        pad5 = self.pad5(dec5)

        cat6 = torch.cat((pad5,enc1),dim=1)
        dec6 = self.dec6(cat6) 
      
        x = torch.tanh(dec6)


        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels,out_channels,nker=64,norm='inorm'):
        super(Discriminator,self).__init__()

        self.enc1 = CBR2d(1 * in_channels, 1 * nker, kernel_size=4,stride=2,
                          padding=1,norm=None,relu=0.2,bias=False)   # 첫번째 D layer에는 batch 적용 X 
        self.enc2 = CBR2d(1 * nker, 2 * nker, kernel_size=4,stride=2,
                          padding=1,norm=norm,relu=0.2,bias=False)
        self.enc3 = CBR2d(2*nker, 4 * nker, kernel_size=4,stride=2,
                          padding=1,norm=norm,relu=0.2,bias=False)
        self.enc4 = CBR2d(4 * nker, 8 * nker, kernel_size=4,stride=2,
                          padding=1,norm=norm,relu=0.2,bias=False)
        self.enc5 = CBR2d(8 * nker, out_channels, kernel_size=4,stride=2,
                          padding=1,norm=None,relu=None,bias=False)

    def forward(self,x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)

        x = torch.sigmoid(x)

        return x
