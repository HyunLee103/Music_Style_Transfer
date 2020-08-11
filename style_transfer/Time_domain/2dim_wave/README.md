# style transfer using 2-D waveform
Attempt to transform style in time domain due to resolution problem in frequency domain. 
To apply the time domain audio representation waveform to GAN, it is used by transforming it into a two-dimensional form(2D waveform).

## CycleGAN
We purely applied the cycle gan first. The resolution of sound was better than the frequency domain, 
but there was a problem with weak style conversion, which was the fundamental problem of cycle gan.

## MelGAN
MelGAN is a model that reflects the structural loss between the input space of the generator and the generative space through the siamese network. However, since it is a model that applies to spectrogram, we concat input one-dimensional vector waveform axially to create a two-dimensional wave. Through this, not only can the melGAN be applied to the waveform, but also the dilation effect can be expected. This model was not satisfied with the result and we decided to try the autoencoder, not the generative model.
