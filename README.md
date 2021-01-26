# Music Style Tranfer
Aim to create an artificial neural network that changes music style, eg) Beatles - hey jude -> hey jude Jazz ver.

## Demo
https://hyunlee103.tistory.com/80

## Dataset
![image](https://user-images.githubusercontent.com/52783941/105868195-457aa500-6039-11eb-989c-d95efbd3d9f2.png)  
Because the resolution of the audio separation was bad, we needed a separate sound source for the instrument. We used MUSDB18(https://github.com/sigsep/sigsep-mus-db) to satisfy this.

## Requirements
- CUDA 10.0  
- python 3.6.10  
- pytorch 1.7.1  
- numpy 1.19.2  
- opencv-python 4.5.1  

## Usage
You can choose between the time domain and the frequency domain. 

    python main.py --data_dir 'your datapath'
    

## Implementation models

We tried three models, one in the time domain and two in the frequency domain.  

### 1. Frequency domain - CycleGAN

We have applied the CycleGAN model, which shows excellent performance in the style transformation of image domain, to the mel-specrogram in a naive manner. In the process of restoring the specrogram to waveform, the sound source resolution was severely degraded. Moreover, the sound source converted through CycleGAN did not change much from the original. We found the reason in the direction back to self due to cycle loss and only consider the pixel-wise loss due to L1 loss, where the specrogram must achieve structural changes before the style can change. Therefore, we tried waveform instead of specrogram and MelGAN instead of CycleGAN.  

### 2. Time domain - MelGAN

MelGAN is a model that reflects the structural loss between the input space of the generator and the generative space through the siamese network. However, since it is a model that applies to spectrogram, we concat input one-dimensional vector waveform axially to create a two-dimensional wave. Through this, not only can the melGAN be applied to the waveform, but also the dilation effect can be expected. This model was not satisfied with the result and we decided to try the autoencoder, not the generative model.

### 3. Time domain - Autoencoder

We tried Universal Music Translation(https://github.com/facebookresearch/music-translation) for style transfer rock to jazz piano. While this paper translates musical instruments versus musical instruments such as violin, cello, and piano, we tried to transfer the whole rock music into jazz piano. Because this model is based on wavenet, learning and inference cost is very high.


## Limits and Future Studies

There is a limit to the application of prior computer vision research due to differences in image and audio data. Due to the high cost of waveNet, it is difficult to increase the resolution of the results and the real-time service seems to be a long way to go. The future direction of research is to identify the data characteristics that affect the music style and create a model that takes those characteristics into account. Also, we need to make low cost models for high-resolution real-time models.


## Reference
- Musdb18 Dataset  
- MUSIC SOURCE SEPARATION USING STACKED HOURGLASS NETWORK(Park et al, 2018 ISMR)  
- Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks  
- WaveNet: A Generative Model for Raw Audio(Deep Mind, 2016)  
- META-LEARNING EXTRACTORS FOR MUSIC SOURCE SEPARATION(Samuel et al. 2020)  
- MelGAN-VC: Voice Conversion and Audio Style Transfer on arbitrarily long samples using Spectrograms(Marco Pasini, 2020)  
- A Universal Music Translation Network(Noam Mor el al, 2018)  


## Contributor 
Kyojung Koo(https://github.com/koo616), Sanghyung Jung(https://github.com/SangHyung-Jung), Hyun Lee


## Citation
    
    @misc{musdb18,
     author       = {Rafii, Zafar and
                  Liutkus, Antoine and
                  Fabian-Robert St{\"o}ter and
                  Mimilakis, Stylianos Ioannis and
                  Bittner, Rachel},
     title        = {The {MUSDB18} corpus for music separation},
     month        = dec,
     year         = 2017,
     doi          = {10.5281/zenodo.1117372},
     url          = {https://doi.org/10.5281/zenodo.1117372} 
    }
