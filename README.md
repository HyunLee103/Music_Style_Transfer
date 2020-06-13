# Music_Style_Transfer
Project repository for music style transfer with Neural Network.(2020.03 ~ )

### Team member
Kyojung Koo, SangHyung Jung, Hyun Lee


# Overview
Aim to create an artificial neural network that changes music style, eg) Beatles-hey jude -> hey jude Jazz ver.

We largely divided the task into three and read the paper corresponding to each task.

## 1. Style transfer

It was the core task of our project, so we searched for the most papers. Most of them used cycleGAN, which is a generic model. The key idea is image-to-image CycleGAN, which utilizes the specrogram (CQT) after passing the sound source to Frequency domain through Fourier transform. What was interesting was that music data was treated like images in the Frequency domain and cycleGAN was applied, and CQT was used instead of specrogram to save the profile information. We read and studied the following papers.
