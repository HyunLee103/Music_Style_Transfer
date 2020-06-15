from layer import *


## UNet + SpectralNorm
"""
enc1, dec4에 batchnorm X, 마지막 dec layer는 activation X -> 마지막에 tanh 통과시키므로
"""
class Generator(nn.Module):
    def __init__(self,input_shape):
        super(Generator, self).__init__()
        
        h, w, c = input_shape

        #downscaling
        self.enc1 = CBR2d(c, 64, kernel_size=4, stride=2, padding=1 ,norm=None,relu=0.2)
        self.enc2 = CBR2d(64, 128, kernel_size=4, stride=2, padding=1 ,norm='bnorm',relu=0.2)
        self.enc3 = CBR2d(128, 256, kernel_size=4, stride=2, padding=1 ,norm='bnorm',relu=0.2)
        self.enc4 = CBR2d(256, 256, kernel_size=4, stride=2, padding=1 ,norm='bnorm',relu=0.2)
        
        #upscaling
        self.dec1 = DECBR2d(256,256,kernel_size=4,padding=1,stride=2,norm='bnorm',relu=0.2)
        self.dec2 = DECBR2d(512,128,kernel_size=4,padding=1,stride=2,norm='bnorm',relu=0.2)
        self.dec3 = DECBR2d(256,64,kernel_size=4,padding=1,stride=2,norm='bnorm',relu=0.2)
        self.dec4 = DECBR2d(128,c,kernel_size=4,padding=1,stride=2,norm=None,relu=None) 
  
    def forward(self, x):
        print(x.shape)
        enc1 = self.enc1(x)
        print(enc1.shape)
        enc2 = self.enc2(enc1)
        print(enc2.shape)
        enc3 = self.enc3(enc2)
        print(enc3.shape)
        enc4 = self.enc4(enc3)
        print(enc4.shape)

        dec1 = self.dec1(enc4)
        print(dec1.shape)
        cat2 = torch.cat((dec1,enc3),dim=1)
        dec2 = self.dec2(cat2)
        print(dec2.shape)
        cat3 = torch.cat((dec2,enc2),dim=1)
        dec3 = self.dec3(cat3)
        print(dec3.shape)
        cat4 = torch.cat((dec3,enc1),dim=1)
        print(cat4.shape)
        dec4 = self.dec4(cat4)
        x = torch.tanh(dec4)
        print(x.shape)
        return x

"""    
S 네트워크 아웃풋으로 배치 당 dim=128 latent vector(OK!), 
근데 padding = 'same'이 안되었음 -> 각각 16, 15 padding 줘서 해결
"""
class Siamese(nn.Module):
    def __init__(self,input_shape):
        super(Siamese, self).__init__()
        
        h, w, c = input_shape
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.batchnorm = nn.BatchNorm2d(num_features=256)

        self.g1 = nn.Conv2d(in_channels=c, out_channels=256, kernel_size=(h,9), stride=1, padding=0)
        self.g2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,9), stride=(1,2),padding=(0,16))
        self.g3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,7), stride=(1,2),padding=(0,15))
        self.g4 = nn.Linear(6144, 128)
            
    def forward(self, x):
        print(x.shape)
        x = self.g1(x)
        x = self.batchnorm(self.leaky(x))
        print(x.shape)
        
        ## g2, g3 에 same padding을 줘야한다.
        print(x.shape)
        x = self.batchnorm(self.leaky(x))


        x = self.g3(x)
        print(x.shape)
        x = self.batchnorm(self.leaky(x))

        x = x.view(x.shape[0],-1)
        print(x.shape)
        x = self.g4(x)
        print(x.shape)
        return x

"""
현재 G(Generator)와 다르게 일반 MelGAN 구조, kernel_size가 특이함
g2, g3 padding = same 이라, 각각 51, 50 zero padding으로 맞춰줌
"""
class Discriminator(nn.Module):
    def __init__(self,input_shape):
        super(Discriminator, self).__init__()
        
        h, w, c = input_shape
        

        self.g1 = CBR2d(1, 512, kernel_size=(h,3), stride=1, padding=0,relu=0.2,norm=None)
        self.g2 = CBR2d(512, 512, kernel_size=(1,9), stride=2, padding=(0,51),relu=0.2,norm=None)
        self.g3 = CBR2d(512, 512, kernel_size=(1,7), stride=2, padding=(0,50),relu=0.2,norm=None)
        self.g4 = DenseSN(input_shape=48128)
        
    def forward(self, x):
        print(x.shape)
        x = self.g1(x)
        print(x.shape)
        x = self.g2(x)
        print(x.shape)
        x = self.g3(x)
        print(x.shape)
        x = self.g4(x.view(x.shape[0],-1))
        print(x.shape)
        return x   

"""
1. SpectralNorm이 적용이 되었나?
2. zero padding 괜찮은가? 저렇게 same size로 유지하려는 이유는?
3. G를 UNet + spectral로 바꿨는데 D는 그대로 MelGAN 구조를 써도 되나?
4. D 최종 actication 뭐써야되지? 일단 tanh 했음, CycleGAN은 sigmoid 인듯?
5. DENSN 해석을 위한 tensor dot 함수 해석 help
"""