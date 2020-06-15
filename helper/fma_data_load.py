import os
import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa
import librosa.display
import ast
import shutil

def load_files():
    os.chdir("G:\\공유 드라이브\\Music_Style_Transform\\fma_metadata")
    genres = load("genres.csv")
    tracks = load("tracks.csv")
    features = load("features.csv")
    echonest = load("echonest.csv")

    return genres, tracks, features, echonest

def load(filename):

    if 'features' in filename:
        return pd.read_csv(filename, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filename, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filename, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filename, index_col=0, header=[0, 1])
        
        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                ('track', 'genres'), ('track', 'genres_all')]
                # ('track', genre_top) 은 ast.literal_eval이 필요가 없음; 원래 []가 아니라 그냥 str임
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                ('album', 'date_created'), ('album', 'date_released'),
                ('artist', 'date_created'), ('artist', 'active_year_begin'),
                ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                'category', categories=SUBSETS, ordered=True)

        COLUMNS = [('track', 'license'), ('artist', 'bio'),
                ('album', 'type'), ('album', 'information')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks

genres, tracks, features, echonest = load_files()

np.testing.assert_array_equal(features.index, tracks.index)
assert echonest.index.isin(tracks.index).all()

medium = tracks[tracks['set', 'subset'] <= 'medium']
small = tracks[tracks['set', 'subset'] <= 'small']

os.chdir(r"G:\공유 드라이브\Music_Style_Transform\fma_small")

def sum_genre(genre, set_nm):
    os.mkdir(genre)
    for tid in set_nm.track[set_nm.track.genre_top==genre].index:
        tid = str(tid)
        while len(tid) < 6:
            tid = "0" + tid
        
        for folder in os.listdir():
            try:
                int(folder) 
                pass
            except: continue
            music_list = os.listdir(os.getcwd() + "\\{}".format(folder))
            if tid+".mp3" in music_list:
                shutil.move(os.getcwd()+"\\{}\\{}.mp3".format(folder, tid), os.getcwd()+"\\{}\\{}.mp3".format(genre, tid))

sum_genre("Pop", small)