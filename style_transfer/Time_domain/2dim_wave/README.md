# style transfer using 2-D waveform
Attempt to transform style in time domain due to resolution problem in frequency domain. 
To apply the time domain audio representation waveform to GAN, it is used by transforming it into a two-dimensional form(2D waveform).

## CycleGAN
We purely applied the cycle gan first. The resolution of sound was better than the frequency domain, 
but there was a problem with weak style conversion, which was the fundamental problem of cycle gan.

## MelGAN
