# Music Style Tranfer
music style transfer with Neural Network
Aim to create an artificial neural network that changes music style,  
eg) Beatles - hey jude -> hey jude Jazz ver.

## Demo
https://hyunlee103.tistory.com/80

## Dataset
![image](https://user-images.githubusercontent.com/52783941/105868195-457aa500-6039-11eb-989c-d95efbd3d9f2.png)  
Because the resolution of the audio separation was bad, we needed a separate sound source for the instrument. We used MUSDB18(https://github.com/sigsep/sigsep-mus-db) to satisfy this.



## 1. Style transfer

It was the core task of our project, so we searched for the most papers. Most of them used cycleGAN, which is a generic model. The key idea is image-to-image CycleGAN, which utilizes the specrogram (CQT) after passing the sound source to Frequency domain through Fourier transform. What was interesting was that music data was treated like images in the Frequency domain and cycleGAN was applied, and CQT was used instead of specrogram to save the profile information. We read and studied the following papers.

1) WaveNet

2) GANSynth: Adversary Neural Audio Synthesis

3) Symbolic muisc genre transfer

4) Timbrepton

5) Universal Music Translation

However, if we transfer style from the Frequency domain, there will be a problem with the construction to be looked at below, and the resoultion of the result will not be guaranteed, so I thought about trying it in time domain. 5) According to the Universal Music Translation paper, style transfer is carried out in time domain. So I set the direction of the project based on this paper.

## 2. Implementation models

We tried three models, one in the time domain and two in the frequency domain.  

### 1. Frequency domain - CycleGAN

We have applied the CycleGAN model, which shows excellent performance in the style transformation of image domain, to the mel-specrogram in a naive manner. In the process of restoring the specrogram to waveform, the sound source resolution was severely degraded. Moreover, the sound source converted through CycleGAN did not change much from the original. We found the reason in the direction back to self due to cycle loss and only consider the pixel-wise loss due to L1 loss, where the specrogram must achieve structural changes before the style can change. Therefore, we tried waveform instead of specrogram and MelGAN instead of CycleGAN.  

### 2. Time domain - MelGAN

MelGAN is a model that reflects the structural loss between the input space of the generator and the generative space through the siamese network. However, since it is a model that applies to spectrogram, we concat input one-dimensional vector waveform axially to create a two-dimensional wave. Through this, not only can the melGAN be applied to the waveform, but also the dilation effect can be expected. This model was not satisfied with the result and we decided to try the autoencoder, not the generative model.

### 3. Time domain - Autoencoder

We tried Universal Music Translation(https://github.com/facebookresearch/music-translation) for style transfer rock to jazz piano. While this paper translates musical instruments versus musical instruments such as violin, cello, and piano, we tried to transfer the whole rock music into jazz piano. Because this model is based on wavenet, learning and inference cost is very high.


## 3. Limits and Future Studies

There is a limit to the application of prior computer vision research due to differences in image and audio data. Due to the high cost of waveNet, it is difficult to increase the resolution of the results and the real-time service seems to be a long way to go. The future direction of research is to identify the data characteristics that affect the music style and create a model that takes those characteristics into account. Also, we need to make low cost models for high-resolution real-time models.


## 4. Audio Source Segmentation

It wasn't the first idea that came out, but I thought it would be reasonable to separate the music into components (drum, bass, vocals, and the rest) and transfer the domain rather than targeting the entire song. We studied three papers published in 2018, 2019 and 2020, respectively.

1) Wave-U-Net (2018)

2) Demucs Deep Extractor for Music Source Separation (2019)

3) Meta-Learning Extractor for Music Source Segmentation (2020)

I conducted the Audio Source Segmentation using the pre-train model, which was released from the best metric-learing thesis, and the results were satisfactory.


## 5. Audio Reconstruction

The first thing I thought of was domain transfer from the Frequency domain, so I needed to rebuild it back to the waveform. The algorithm used here mainly uses Griffin-Lim, which was published in 1984. However, I studied the following paper for better results in the solution criteria.

1) Signal Evaluation from Modified Short-time Fourier Transform (Griffin-Lim)

2) WaveNet

3) Deep Griffin Lim

2)WaveNet is an un-supervised model that generates waveform in time domain. It is applied in various fields, and in the 4)Timbretron paper, it was used instead of griffin-lim in the audio reconstruction, which was considered above, and it was said that the performance was better. 5) In Universal Music Translation, Auto-encoder is created using WaveNet.

As mentioned above, Audio reconstruction is necessary for transferring in the Frequency domain, and it is unnecessary because the waveform form is maintained in the time domain.  


## 6. Dataset
Because the resolution of the audio separation was bad, we needed a separate sound source for the instrument. We used MUSDB18(https://github.com/sigsep/sigsep-mus-db) to satisfy this.


### Team member
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
