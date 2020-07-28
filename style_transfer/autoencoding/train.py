from audio_encoding import WaveNet, Encoder, ZDiscriminator
from data import WaveData
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

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

# data load
data_path = r"C:/Users/koo/Desktop/"

wavedata = WaveData(A=data_path + "piano_cover.npy",
                    B=data_path + "musdb_sample.npy")

dataloader = wavedata.get_loader(args["batch_size"],
                                 shuffle=True)

# models
encoder = Encoder(args).cuda()
decoder_A = WaveNet(args).cuda()
decoder_B = WaveNet(args).cuda()
z_discr = ZDiscriminator(args).cuda()

# loss & optim
#criterion = nn.CrossEntropyLoss().cuda()

encoder_optim = torch.optim.Adam(encoder.parameters(), lr=args["lr"])
z_discr_optim = torch.optim.Adam(z_discr.parameters(), lr=args["lr"])
A_optim = torch.optim.Adam(decoder_A.parameters(), lr=args["lr"])
B_optim = torch.optim.Adam(decoder_B.parameters(), lr=args["lr"])

# train
for epoch in range(1, args["n_epochs"]+1):
    print(f"Epoch {epoch}")
    for i, batch in enumerate(dataloader):
        A = batch['A'].cuda()
        B = batch['B'].cuda()
        A_aug = batch['A_aug'].cuda()
        B_aug = batch['B_aug'].cuda()
        
        # z_dicr target
        zeros = torch.zeros(A.size(0)).long().cuda()
        ones = torch.ones(A.size(0)).long().cuda()
 
        ### train z_discr to A ###
        encoder.zero_grad()
        z_discr.zero_grad()
        
        cond_A = encoder(A.float())
        discr_A = z_discr(cond_A)
        
        # get loss
        loss_A = F.cross_entropy(discr_A, zeros)
        loss_discr = loss_A * args['d_lambda']
        # backward
        loss_discr.backward()
        encoder_optim.step()
        z_discr_optim.step()
        
        ### train decoder A ###
        encoder.zero_grad()
        decoder_A.zero_grad()
        
        cond_A = encoder(A_aug.float())
        discr_A = z_discr(cond_A)
        out_A = decoder_A(A.float(), cond_A)  # (batch, 256, seq)
        
        out_A = out_A.transpose(1, 2).contiguous().view(-1, 256)  # (batch * seq, 256)
        target_A = A.view(-1).long()  # (batch * seq)
        # get loss A
        loss_discr = F.cross_entropy(discr_A, zeros)
        loss_decoder = F.cross_entropy(out_A, target_A, reduction='none')  # (batch * seq)
        loss_decoder = loss_decoder.reshape(A.size(0), A.size(1)).mean(1).mean()  # (batch, seq) -> (batch) -> scalar
        loss_A_decoder = loss_decoder - loss_discr * args['d_lambda']
        # backward & step
        loss_A_decoder.backward()
        encoder_optim.step()
        A_optim.step()
        
        ### train z_discr to B ###
        encoder.zero_grad()
        z_discr.zero_grad()   
        
        cond_B = encoder(B.float())
        discr_B = z_discr(cond_B)
        # get loss
        loss_B = F.cross_entropy(discr_B, ones)
        loss_discr = loss_B * args['d_lambda']
        # backward
        loss_discr.backward()
        encoder_optim.step()
        z_discr_optim.step()
        
        ### train decoder B ###
        encoder.zero_grad()
        decoder_B.zero_grad()
        
        cond_B = encoder(B_aug.float())
        discr_B = z_discr(cond_B)
        out_B = decoder_B(B.float(), cond_B)  # (batch, 256, seq)
        
        out_B = out_B.transpose(1, 2).contiguous().view(-1, 256)  # (batch * seq, 256)
        target_B = B.view(-1).long()  # (batch * seq)
        # get loss B
        loss_discr = F.cross_entropy(discr_B, ones)
        loss_decoder = F.cross_entropy(out_B, target_B, reduction='none')  # (batch * seq)
        loss_decoder = loss_decoder.reshape(B.size(0), B.size(1)).mean(1).mean()  # (batch, seq) -> (batch) -> scalar
        loss_B_decoder = loss_decoder - loss_discr * args['d_lambda']
        # backward & step
        loss_B_decoder.backward()
        encoder_optim.step()
        B_optim.step()
        
        print(f"Batch {i+1}/{len(dataloader)}")
        print(f"- [Discriminator] {loss_discr.item():0.4} / [A] {loss_A_decoder.item():0.4} / [B] {loss_B_decoder.item():0.4}")
        
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder_A': decoder_A.state_dict(),
        'decoder_B': decoder_B.state_dict(),
        'z_discr': z_discr.state_dict(),
        'encoder_optim': encoder_optim.state_dict(),
        'decoder_A_optim': A_optim.state_dict(),
        'decoder_B_optim': B_optim.state_dict(),
        'z_discr_optim': z_discr_optim.state_dict(),
    }, data_path + f"{epoch}.pth")