# Hi. I'm Koo.

# FMA dataset
## fma_small
similar with GTZAN dataset which AI audio youtuber uses.  

# A Universal Music Translation Network
## link
https://research.fb.com/publications/a-universal-music-translation-network/  
github: https://github.com/facebookresearch/music-translation 

## Dataset - MusicNet
### link: https://homes.cs.washington.edu/~thickstn/musicnet.html  
### purpose  
1. Identify the notes performed at specific times in a recording.  
2. Classify the instruments that perform in a recording.  
3. Classify the composer of a recording.  
4. Identify precise onset times of the notes in a recording.  
5. Predict the next note in a recording, conditioned on history.

### Data size
MusicNet (Raw - recommended) - Raw (wav, csv) distribution of MusicNet (10.3GB).  
MusicNet (Python) - NumPy distribution of MusicNet (11GB).  
MusicNet (HDF5) - HDF5 distribution of MusicNet (7.1GB).  

### Only classical instruments (ex. violin, piano etc.)
We want to seperate band sound music.  
So, ignore it. bye!  

### Model
using Wavenet  
![model](https://github.com/HyunLee103/Music_Style_Transfer/blob/master/gu/fig/A%20Universal%20Music%20Translation%20Network%20-%20model.png?raw=true)
- Shared encoder + independant decoder + source discriminator  
