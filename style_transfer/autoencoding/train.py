from audio_encoding import WaveNet, Encoder, ZDiscriminator
from data import WaveData
import torch
import torch.nn.functional as F
from itertools import chain

# origin
args = {
    # encoder
    "encoder_blocks": 3,
    "encoder_layers": 10,
    "encoder_channels": 128,
    "latent_d": 64,
    "encoder_func": 'relu',
    
    # decoder
    "blocks": 4,
    "layers": 14,
    "kernel_size": 2, 
    "skip_channels": 128,
    "residual_channels": 128,
    
    # discriminator
    "n_datasets": 4,
    "d_layers": 3,
    "d_channels": 100,
    "d_lambda": 0.01,
    "dropout_rate": 0.0,
    
    # train
    "n_epochs": 300,
    "batch_size": 4,
    "lr": 0.001,
    "load_epochs": 43
}

# custom for batch 8
# args = {
#     # encoder
#     "encoder_blocks": 2,
#     "encoder_layers": 10,
#     "encoder_channels": 128,
#     "latent_d": 64,
#     "encoder_func": 'relu',
    
#     # decoder
#     "blocks": 2,
#     "layers": 10,
#     "kernel_size": 2, 
#     "skip_channels": 128,
#     "residual_channels": 128,
    
#     # discriminator
#     "n_datasets": 2,
#     "d_layers": 3,
#     "d_channels": 100,
#     "d_lambda": 0.01,
#     "dropout_rate": 0.0,
    
#     # train
#     "n_epochs": 15,
#     "batch_size": 8,
#     "lr": 0.001
# }

# data load
data_path = r"/home/chdnjf103/"

wavedata = WaveData([data_path + "piano.npy",
                     data_path + "musdb_other.npy",
                     data_path + "musdb_vocal.npy",
                     data_path + "musdb_bass.npy"])

dataloader = wavedata.get_loader(args["batch_size"],
                                 shuffle=True)

# models
encoder = Encoder(args).cuda()
decoders = [WaveNet(args).cuda() for _ in range(args['n_datasets'])]
z_discr = ZDiscriminator(args).cuda()

# optim
z_discr_optim = torch.optim.Adam(z_discr.parameters(), lr=args["lr"])
optims = [torch.optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=args['lr']) \
          for decoder in decoders]

# pth
bestmodel = torch.load(data_path + f"models/{args['load_epochs']}.pth")
encoder.load_state_dict(bestmodel['encoder'])
for i in range(decoders):
    decoders[i].load_state_dict(bestmodel['decoders'][i])
    optims[i].load_state_dict(bestmodel["decoder_optims"][i])
z_discr.load_state_dict(bestmodel['z_discr'])
z_discr_optim.load_state_dict(bestmodel['z_discr_optim'])

def train_discr(encoder, z_discr, z_discr_optim, x_aug, datanum):
    ### train z_discr to A ###
    z_discr_optim.zero_grad()
    # forward
    cond = encoder(x_aug.float())
    discr = z_discr(cond.detach())
    # get loss
    loss = F.cross_entropy(discr, torch.tensor([datanum]*x_aug.size(0)).long().cuda())
    loss_discr = loss * args['d_lambda']
    # backward & step
    loss_discr.backward()
    z_discr_optim.step()
    
    return loss_discr.item()
    
def train_decoder(encoder, z_discr, decoder, decoder_optim,
                  x, x_aug, datanum):
    ### train decoder A ###
    decoder_optim.zero_grad()
    # forward
    cond = encoder(x_aug.float())
    discr = z_discr(cond)
    out = decoder(x.float(), cond)  # (batch, 256, seq)
    # manipulate for cross entropy
    out = out.transpose(1, 2).contiguous().view(-1, 256)  # (batch * seq, 256)
    target = x.view(-1).long()  # (batch * seq)
    # get loss A
    loss_discr = F.cross_entropy(discr, torch.tensor([datanum]*x.size(0)).long().cuda())
    loss_recon = F.cross_entropy(out, target, reduction='none')  # (batch * seq)
    loss_recon = loss_recon.reshape(x.size(0), x.size(1)).mean(1).mean()  # (batch, seq) -> (batch) -> scalar
    loss_decoder = loss_recon - loss_discr * args['d_lambda']
    # backward & step
    loss_decoder.backward()
    decoder_optim.step()
    
    return loss_decoder.item()

# train
for epoch in range(args['load_epochs'] + 1, args["n_epochs"] + args['load_epochs'] + 1):
    print(f"\nEpoch {epoch}")
    for i, (x, x_aug) in enumerate(dataloader):
        print(f"\nBatch {i+1}/{len(dataloader)}")
        for datanum in range(args['n_datasets']):
            d_loss = train_discr(encoder, z_discr, z_discr_optim,
                        x_aug[datanum].cuda(), datanum)
            
            loss = train_decoder(encoder, z_discr, decoders[datanum], optims[datanum],
                          x[datanum].cuda(), x_aug[datanum].cuda(), datanum)
            
            print(f"[{datanum}] {d_loss:0.4}, {loss:0.4}", end='  ')
    
    torch.save({
        'encoder': encoder.state_dict(),
        'decoders': [decoder.state_dict() for decoder in decoders],
        'z_discr': z_discr.state_dict(),
        'decoder_optims': [optim.state_dict() for optim in optims],
        'z_discr_optim': z_discr_optim.state_dict()
    }, data_path + f"models/{epoch}.pth")