from audio_encoding import WaveNet, WavenetGenerator, Encoder
from utils import mu_law, inv_mu_law
import torch
import librosa

args = {
    # encoder
    "encoder_blocks": 2,
    "encoder_layers": 10,
    "encoder_channels": 128,
    "latent_d": 64,
    "encoder_func": 'relu',
    
    # decoder
    "blocks": 2,
    "layers": 10,
    "kernel_size": 2, 
    "skip_channels": 128,
    "residual_channels": 128,
    
    # discriminator
    "n_datasets": 2,
    "d_layers": 3,
    "d_channels": 100,
    "d_lambda": 0.01,
    "dropout_rate": 0.0,
    
    # train
    "n_epochs": 15,
    "batch_size": 8,
    "lr": 0.001
}
data_path = r"C:/Users/koo/Desktop/"

# models
encoder = Encoder(args).cuda()
decoder = WaveNet(args).cuda()

# load state dict
pth = torch.load(data_path + "4.pth")
encoder.load_state_dict(pth['encoder'])
decoder.load_state_dict(pth['decoder_A'])

# load sample & mu_law
sample, sr = librosa.core.load(data_path + "위잉위잉.mp3",sr=16000)
sample_part = mu_law(sample[:16000*4])
sample_tensor = torch.tensor(sample_part).unsqueeze(0).cuda()

# encoding
with torch.no_grad():
    cond = encoder(sample_tensor.float())
    
# generator
generator = WavenetGenerator(decoder)
    
# 20 * 800(cond_repeat) = 16000 -> 1 second
cond_split = torch.split(cond, 20, -1)

audio_out = []
for cond_chunk in cond_split:
    audio_out.append(generator.generate(cond_chunk).cpu())
    
out = torch.cat(audio_out, dim=-1)
out = inv_mu_law(out.numpy()[0][0])
