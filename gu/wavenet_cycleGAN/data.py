import librosa


def processing_for_wavenet(data):
    """
    data: -1 ~ 1 float64 waveform
    
    return: 0 ~ 255 int
    """
    
    quant = librosa.core.mu_compress(data, mu=255, quantize=True) # -127 ~ 128
    quant += 127  # 0 ~ 255
    
    assert 'int' in quant.dtype, "Not int. check mu_compress function"
    
    return quant
    