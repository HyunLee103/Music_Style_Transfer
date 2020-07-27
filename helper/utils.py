import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pytube import YouTube
import glob, os
import pandas as pd

def convert():
    ## mp4 파일을 wav 파일로 convert 해주는 함수 ( url 다운로드 끝나면 실행 )
    files = glob.glob("*.mp4")
    for x in files:
        if not os.path.isdir(x):
            filename = os.path.splitext(x)
            try:
                os.rename(x, filename[0] + ".wav")
            except:
                pass


def youtube_download(url):
    ## youtube에서 128kbps 기준으로 audio 파일을 다운로드 하는데 mp4로 다운로드됨
    yt = YouTube(url)
    yt.streams.filter(only_audio=True, abr='128kbps').first().download()

def stft_spectogram(wav, sr, hop_length=512, n_fft=2048):
    signal, sr = librosa.load(wav, sr=sr)
    stft = librosa.core.stft(signal, hop_length = hop_length, n_fft = n_fft)
    spectogram = np.abs(stft)
    log_spectogram = librosa.amplitude_to_db(spectogram)

    return log_spectogram

def mfccs_spectogram(wav, sr, hop_length=512, n_fft=2048):
    signal, sr = librosa.load(wav, sr=sr)
    mfccs = librosa.feature.mfcc(signal, n_ftt=n_fft, hop_length=hop_length)
    
    return mfccs

def show_spectogram(spectogram, sr, hop_length=512):
    librosa.display.specshow(spectogram, sr = sr, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()
    plt.show()
