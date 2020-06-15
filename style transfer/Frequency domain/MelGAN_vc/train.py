## Define Loss
def mae(x,y):
    return torch.mean(torch.abs(x - y))

def mse(x,y):
    return torch.mean((x-y)**2)

def loss_travel(sa,sab,sa1,sab1):
    #cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    #l1 = cos((sa-sa1), (sab-sab1))
    #print("l1 shape :", l1.shape)
    #l2 = F.normalize((sa-sa1) - (sab-sab1), p=2).mean(axis=1)
    #print("l2 shape: ", l2.shape)
    l1 = torch.mean(((sa-sa1)-(sab-sab1))**2)
    l2 = torch.mean(torch.sum(-(F.normalize(sa-sa1, p=2, dim=-1) * 
            F.normalize(sab-sab1, dim=-1)), dim=-1))
    return l1 + l2

def loss_siamese(sa,sa1):
    logits = torch.sqrt(torch.sum(((sa-sa1)**2), axis=-1, keepdim=True))
    return Variable(torch.mean(torch.max((delta - logits), 0)[0]**2), requires_grad=True)

def d_loss_f(fake):
    return Variable(torch.mean(torch.max(1 + fake, 0)[0]), requires_grad=True)

def d_loss_r(real):
    return Variable(torch.mean(torch.max(1 - real, 0)[0]), requires_grad=True)

def g_loss_f(fake):
    return torch.mean(-fake)

## optimizer, model 설정

G = Generator((hop,48,1)).to(device)
S = Siamese((hop,48,1)).to(device)
D = Discriminator((hop,96,1)).to(device)

#Generator loss is a function of 
params = list(G.parameters()) + list(S.parameters())
opt_gen = optim.Adam(params, lr=1e-5)

opt_disc = optim.SGD(D.parameters(), lr=1e-5)

#Set learning rate
def update_lr(gen_lr, dis_lr):
    opt_gen.lr = gen_lr
    opt_disc.lr = dis_lr

## train_All

def train_all(a,b):
    #splitting spectrogram in 3 parts
    aa,aa2 = extract_image(a) 
    bb,bb2 = extract_image(b)

    #gen.zero_grad()
    #critic.zero_grad()
    #siam.zero_grad()
    
    opt_gen.zero_grad()

    #translating A to B
    fab = G.forward(aa)
    fab2 = G.forward(aa2)
    
    #identity mapping B to B  COMMENT THESE 3 LINES IF THE IDENTITY LOSS TERM IS NOT NEEDED
    #fid = gen.forward(bb)
    #fid2 = gen.forward(bb2)
    #fid3 = gen.forward(bb3)
    
    #concatenate/assemble converted spectrograms
    fabtot = assemble_image([fab,fab2])

    #feed concatenated spectrograms to critic
    cab = D.forward(fabtot) # D(G(x))
    cb = D.forward(b)       # D(y)

    #feed 2 pairs (A,G(A)) extracted spectrograms to Siamese
    sab = S.forward(fab)
    sab2 = S.forward(fab2)
    sa = S.forward(aa)
    sa2 = S.forward(aa2)

    #identity mapping loss
    #loss_id = (mae(bb,fid)+mae(bb2,fid2)+mae(bb3,fid3))/3.      #loss_id = 0. IF THE IDENTITY LOSS TERM IS NOT NEEDED
    #travel loss
    loss_travel_temp = loss_travel(sa,sab,sa2,sab2)
    loss_siamese_temp = loss_siamese(sa,sa2)
    loss_m = loss_travel_temp + loss_siamese_temp
    print("Loss m: ", loss_m)
    #get gen and siam loss and bptt
    loss_g = g_loss_f(cab)
    print("Loss g: ", loss_g)
    lossgtot = loss_g+10.*loss_m #+0.5*loss_id #CHANGE LOSS WEIGHTS HERE  (COMMENT OUT +w*loss_id IF THE IDENTITY LOSS TERM IS NOT NEEDED)

    lossgtot.backward()  # G, S 로 backpropa
    #for idx, i in enumerate(params):
    #    print("Param ", idx, ":")
    #    print(i.grad)
    opt_gen.step()       # G, S update
    
    #get critic loss and bptt
    opt_disc.zero_grad()

    loss_dr = d_loss_r(cb) 
    loss_df = d_loss_f(cab) 
    loss_d = (loss_dr+loss_df)/2.

    print("Loss d: ", loss_d)
    loss_d.backward()
    #for idx, i in enumerate(critic.parameters()):
    #    print("Param ", idx, ":")
    #    print(i.grad)
    opt_disc.step()
    
    return loss_dr,loss_df,loss_g,loss_d

## train D

def train_d(a,b):
    opt_disc.zero_grad()
    
    aa,aa2, = extract_image(a)
    
    #translating A to B
    fab = G.forward(aa)
    fab2 = G.forward(aa2)
    #concatenate/assemble converted spectrograms
    fabtot = assemble_image([fab,fab2])

    #feed concatenated spectrograms to critic
    cab = D.forward(fabtot)
    cb = D.forward(b)

    #get critic loss and bptt
    loss_dr = d_loss_r(cb)
    loss_df = d_loss_f(cab)
    loss_d = (loss_dr+loss_df)/2.
    # loss_d = torch.autograd.Variable(loss_d, requires_grad = True)
    
    loss_d.backward()
    opt_disc.step()

    return loss_dr,loss_df

## 최종 train

def train(epochs, batch_size=16, gen_lr=1e-5, dis_lr=1e-5, n_save=6, gupt=5):
  
    update_lr(gen_lr, dis_lr)
    df_list = []
    dr_list = []
    g_list = []
    id_list = []
    c = 0
    g = 0
  
    for epoch in range(epochs):
        for batchi,(a,b) in enumerate(zip(a_loader,b_loader)):
            #only train discriminator every gupt'th batch
            if batchi%gupt==0:
                dloss_t,dloss_f,gloss,idloss = train_all(a,b)
            else:
                dloss_t,dloss_f = train_d(a,b)

            df_list.append(dloss_f)
            dr_list.append(dloss_t)
            g_list.append(gloss)
            id_list.append(idloss)
            c += 1
            g += 1

            if batchi%600==0:
                # print(f'[Epoch {epoch}/{epochs}] [Batch {batchi}] [D loss f: {np.mean(df_list[-g:], axis=0)} ', end='')
                # print(f'r: {np.mean(dr_list[-g:], axis=0)}] ', end='')
                # print(f'[G loss: {np.mean(g_list[-g:], axis=0)}] ', end='')
                # print(f'[ID loss: {np.mean(id_list[-g:])}] ', end='')
                # print(f'[LR: {lr}]')
                g = 0
            nbatch=batchi

        c = 0