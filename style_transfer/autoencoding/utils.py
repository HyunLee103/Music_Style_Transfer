import numpy as np
import random
import librosa
import IPython

def mu_law(x, mu=255):
    x = np.clip(x, -1, 1)
    x_mu = np.sign(x) * np.log(1 + mu*np.abs(x))/np.log(1 + mu)
    return ((x_mu + 1)/2 * mu).astype('int16')


def inv_mu_law(x, mu=255.0):
    x = np.array(x).astype(np.float32)
    y = 2. * (x - (mu+1.)/2.) / (mu+1.)
    return np.sign(y) * (1./mu) * ((1. + mu)**np.abs(y) - 1.)


def wave_augmentation(wav, wav_freq=16000, magnitude=0.5):
    length = wav.shape[0]
    perturb_length = random.randint(length // 4, length // 2)
    perturb_start = random.randint(0, length // 2)
    perturb_end = perturb_start + perturb_length
    pitch_perturb = (np.random.rand() - 0.5) * 2 * magnitude

    ret = np.concatenate([wav[:perturb_start],
                            librosa.effects.pitch_shift(wav[perturb_start:perturb_end],
                                                        wav_freq, pitch_perturb),
                            wav[perturb_end:]])

    return ret

def play_music(mus, sr=16000):
    return IPython.display.display(IPython.display.Audio(mus, rate=sr))

