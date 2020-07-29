import librosa, librosa.display # recommend librosa == 0.7.2
import numpy as np
import glob, os
import pandas as pd
import random
import musdb
from utils import *

"""
sampling music for musdb and youtube(musdb w/o drum)
you need to import and load musdb root dataset

duraion : sample duration(int)
audio : only for youtube wav(np.array from librosa.load)
rate : sample rate
mode : musdb or youtube

"""
# load musdb
mus = musdb.DB(root="/content/drive/My Drive/ADV_Project_Music_style_transform/new_dataset/musdb18") # root dataset
mus_7 = musdb.DB(download=True)

# musdb rock/band sound track list
required_track = [4,5, 6, 7,11, 12, 14, 15, 17, 23, 24, 25, 27, 29, 30, 37, 38, 42, 43, 44, 46, 48, 52, 54, 55, 56, 57, 59, 60, 62, 63, 64, 65, 67, 71, 74, 77, 78, 80, 81, 85, 86, 91, 92, 93, 96, 98, 99, 101, 104, 107]

# mk required_track_name
required_track_name = []
for i,track in enumerate(mus_7):
    if int(i) in required_track:
        required_track_name.append(track.name)


def db(wave):
    return 20*np.log10(np.sqrt((wave**2).mean()))


def sample_music(duration,audio,rate,inst,mode='musdb'):
    if mode == 'musdb':
        for track in tqdm(mus):
            if track.name in required_track_name:
                if inst == 'vocals':
                    sig = (track.targets['vocals'].audio).T
                elif inst == 'other':
                    sig = (track.targets['other'].audio).T
                elif inst == 'bass':
                    sig = (track.targets['bass'].audio).T

                sig = librosa.resample(sig,44100,rate)
                sig = librosa.core.to_mono(sig)

                for d in range(int(track.duration-10)):
                    start = (d+10) * duration * rate
                    stop = start + (duration*rate)
                    result = sig[int(start):int(stop)]
      
                    if db(result) < -25: continue

                    result = result[:,np.newaxis]

                    try:
                        cat = np.concatenate((cat,result),axis=1)
                    except:
                        cat = result
        return cat.T
    elif mode =='youtube':
        length = int(len(audio)/rate)-200 # -200, 끝부분 소리 X
        iter = int(length/duration)
        for d in tqdm(range(iter)):
            start = (d+1) * duration * rate
            stop = start + (duration*rate)
            result = audio[int(start):int(stop)]

            if db(result) < -25: continue

            result = result[:,np.newaxis]
            try:
                cat = np.concatenate((cat,result),axis=1)
            except:
                cat = result
        return cat.T



"""
sample music for inference 
"""

def sample_test(path,sr,duration):
    audio, rate = librosa.load(path,sr=sr)
    for d in tqdm(range(int(len(audio)/(sr*duration)))):
        start = d * duration * sr
        stop = start + (duration*sr)
        result = audio[int(start):int(stop)]
        result = result[:,np.newaxis]

        try:
            cat = np.concatenate((cat,result),axis=1)
        except:
            cat = result
        
    return cat.T