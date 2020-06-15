
Music_Style_TransferProject repository for music style transfer with Neural Network.(2020.03 ~ )

### Team member
Kyojung Koo(https://github.com/koo616) ,Sanghyung Jung(https://github.com/SangHyung-Jung), Hyun Lee


# Overview
Aim to create an artificial neural network that changes music style, eg) Beatles-hey jude -> hey jude Jazz ver.

We largely divided the task into three and read the paper corresponding to each task.

## 1. Style transfer

It was the core task of our project, so we searched for the most papers. Most of them used cycleGAN, which is a generic model. The key idea is image-to-image CycleGAN, which utilizes the specrogram (CQT) after passing the sound source to Frequency domain through Fourier transform. What was interesting was that music data was treated like images in the Frequency domain and cycleGAN was applied, and CQT was used instead of specrogram to save the profile information. We read and studied the following papers.

1) WaveNet

2) GANSynth: Adversary Neural Audio Synthesis

3) Symbolic muisc genre transfer

4) Timbrepton

5) Universal Music Translation

However, if we transfer style from the Frequency domain, there will be a problem with the construction 3) to be looked at below, and the resoultion of the result will not be guaranteed, so I thought about trying it in time domain. 5) According to the Universal Music Translation paper, style transfer is carried out in time domain. So I set the direction of the project based on this paper.

## 2. Audio Source Segmentation

It wasn't the first idea that came out, but I thought it would be reasonable to separate the music into components (drum, bass, vocals, and the rest) and transfer the domain rather than targeting the entire song. We studied three papers published in 2018, 2019 and 2020, respectively.

1) Wave-U-Net (2018)

2) Demucs Deep Extractor for Music Source Separation (2019)

3) Meta-Learning Extractor for Music Source Segmentation (2020)

I conducted the Audio Source Segmentation using the pre-train model, which was released from the best metric-learing thesis, and the results were satisfactory.

## 3. Audio Reconstruction

The first thing I thought of was domain transfer from the Frequency domain, so I needed to rebuild it back to the waveform. The algorithm used here mainly uses Griffin-Lim, which was published in 1984. However, I studied the following paper for better results in the solution criteria.

1) Signal Evaluation from Modified Short-time Fourier Transform (Gripin-Lim)

2) WaveNet

3) Deep Griffin Lim

2)WaveNet is an un-supervised model that generates waveform in time domain. It is applied in various fields, and in the 4)Timbretron paper, it was used instead of griffin-lim in the audio reconstruction, which was considered above, and it was said that the performance was better. 5) In Universal Music Translation, Auto-encoder is created using WaveNet.

As mentioned above, Audio reconstruction is necessary for transferring in the Frequency domain, and it is unnecessary because the waveform form is maintained in the time domain.
