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
        self.enc1 = CBR2d(1, 64, kernel_size=4, stride=2, padding=1 ,norm=None,relu=0.2)
        self.enc2 = CBR2d(64, 128, kernel_size=4, stride=2, padding=1 ,norm='inorm',relu=0.2)
        self.enc3 = CBR2d(128, 256, kernel_size=4, stride=2, padding=1 ,norm='inorm',relu=0.2)
        self.enc4 = CBR2d(256, 512, kernel_size=4, stride=2, padding=1 ,norm='inorm',relu=0.2)
        self.enc5 = CBR2d(512, 512, kernel_size=4, stride=2, padding=1 ,norm='inorm',relu=0.2)
        self.enc6 = CBR2d(512, 512, kernel_size=4, stride=2, padding=1 ,norm='inorm',relu=0.2)
        
        #upscaling
        self.dec1 = DECBR2d(512,512,kernel_size=4,padding=1,stride=2,norm='inorm',relu=0.2)
        self.dec2 = DECBR2d(1024,512,kernel_size=4,padding=1,stride=2,norm='inorm',relu=0.2)
        self.dec3 = DECBR2d(1024,256,kernel_size=4,padding=1,stride=2,norm='inorm',relu=0.2)
        self.dec4 = DECBR2d(512,128,kernel_size=4,padding=1,stride=2,norm='inorm',relu=0.2) 
        self.dec5 = DECBR2d(256,64,kernel_size=4,padding=1,stride=2,norm='inorm',relu=0.2) 
        self.dec6 = DECBR2d(128,1,kernel_size=4,padding=1,stride=2,norm=None,relu=None) 
  
    def forward(self, x):
        # print(x.shape)
        enc1 = self.enc1(x)
        # print(enc1.shape)
        enc2 = self.enc2(enc1)
        # print(enc2.shape)
        enc3 = self.enc3(enc2)
        # print(enc3.shape)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        # print(enc4.shape)

        dec1 = self.dec1(enc6)
        dec2 = self.dec2(torch.cat([enc5, dec1], dim=1))
        dec3 = self.dec3(torch.cat([enc4, dec2], dim=1))
        dec4 = self.dec4(torch.cat([enc3, dec3], dim=1))
        dec5 = self.dec5(torch.cat([enc2, dec4], dim=1))
        dec6 = self.dec6(torch.cat([enc1, dec5], dim=1))
        # print(x.shape)
        # x = torch.tanh(x)
        return x


class Siamese(nn.Module):
    def __init__(self,input_shape):
        super(Siamese, self).__init__()
        
        h, w, c = input_shape
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.batchnorm = nn.BatchNorm2d(num_features=256)

        self.g1 = CBR2d(1, 64, kernel_size=4, stride=2, padding=1 ,norm='bnorm',relu=0.2)
        self.g2 = CBR2d(64, 128, kernel_size=4, stride=2, padding=1 ,norm='bnorm',relu=0.2)
        self.g3 = CBR2d(128, 256, kernel_size=4, stride=2, padding=1 ,norm='bnorm',relu=0.2)
        self.g4 = nn.Linear(65536, 128)
            
    def forward(self, x):
        # print(x.shape)
        x = self.g1(x)
        x = self.g2(x)
        x = self.g3(x)
        x = x.view(x.shape[0],-1)
        x = self.g4(x)
        x= torch.sigmoid(x)
        return x

"""
현재 G(Generator)와 다르게 일반 MelGAN 구조, kernel_size가 특이함
g2, g3 padding = same 이라, 각각 51, 50 zero padding으로 맞춰줌

-> padding이 너무 많고 kernel_size가 특이해서 일반적인 discriminator 구조 차용
"""
class Discriminator(nn.Module):
    def __init__(self,input_shape):
        super(Discriminator, self).__init__()
        
        h, w, c = input_shape
        

        self.g1 = CBR2d(1, 64, kernel_size=4, stride=2, padding=1 ,norm='snorm',relu=0.2)
        self.g2 = CBR2d(64, 128, kernel_size=4, stride=2, padding=1 ,norm='snorm',relu=0.2)
        self.g3 = CBR2d(128, 256, kernel_size=4, stride=2, padding=1 ,norm='snorm',relu=0.2)
        self.g4 = CBR2d(256, 1, kernel_size=4, stride=2, padding=1 ,norm='snorm',relu=None)

        
    def forward(self, x):
        # print(x.shape)
        x = self.g1(x)
        # print(x.shape)
        x = self.g2(x)
        # print(x.shape)
        x = self.g3(x)
        # print(x.shape)
        # x = self.g4(x.view(x.shape[0],-1))
        x = self.g4(x)
        # print(x.shape)
        # x = torch.tanh(x)
        return x

"""
1. SpectralNorm이 적용이 되었나? -> yes

2. zero padding 괜찮은가? 저렇게 same size로 유지하려는 이유는? -> 해보고, kernel_size랑 padding(zero, reflect) 같이 고려해봐야 될 듯 

3. G를 UNet + spectral로 바꿨는데 D는 그대로 MelGAN 구조를 써도 되나 -> 구조보다는 G, D의 복잡도가 중요. 왜냐면 한 쪽이 너무 강하게 학습되면 안되기 때문. # of parameters, layers/ 상대적으로 D보다는 G를 강력하게 만드는게 좋음. D가 학습되기 쉽기 때문(이진 분류).

4. D 최종 activation 뭐써야되지? 일단 tanh 했음, CycleGAN은 sigmoid 인듯? -> 상관없다

6. SpectralNorm - time domain에 cycleGAN에도 적용해봐도? -> yes

7. 96으로 split하는거 괜찮나.. 1초도 안되는것 같은데 거기서 어떤걸 잡을 수 가 있나 -> 해보고 split 범위를 넓혀보자

8. gtzan이랑 melon 섞어서 셔플? -> yes

9. generator 아웃풋 == D 인풋(구조랑 차원)
"""