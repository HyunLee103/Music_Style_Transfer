from model import *
from dataset import *

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


#functions to be written here
def train_all(a,b):
    #splitting spectrogram in 3 parts
    a1,a2,a3 = extract_image(a) 
    
    b1,b2,b3 = extract_image(b)

    #gen.zero_grad()
    #critic.zero_grad()
    #siam.zero_grad()
    
    optimG.zero_grad()

    #translating A to B
    g_a1 = netG.forward(a1)
    g_a2 = netG.forward(a2)
    g_a3 = netG.forward(a3)
    
    #identity mapping B to B  COMMENT THESE 3 LINES IF THE IDENTITY LOSS TERM IS NOT NEEDED
    #fid = gen.forward(bb)
    #fid2 = gen.forward(bb2)
    #fid3 = gen.forward(bb3)
    
    #concatenate/assemble converted spectrograms
    g_a_total = assemble_image([g_a1,g_a2,g_a3])

    #feed concatenated spectrograms to critic
    fake = netD.forward(g_a_total)
    real = netD.forward(b)

    #feed 2 pairs (A,G(A)) extracted spectrograms to Siamese
    s_ga1 = netS.forward(g_a1)
    s_ga2 = netS.forward(g_a3)
    s_a1 = netS.forward(a1)
    s_a2 = netS.forward(a3)

    #identity mapping loss
    #loss_id = (mae(bb,fid)+mae(bb2,fid2)+mae(bb3,fid3))/3.      #loss_id = 0. IF THE IDENTITY LOSS TERM IS NOT NEEDED
    #travel loss
    loss_travel_temp = loss_travel(s_a1,s_ga1,s_a2,s_ga2)
    loss_siamese_temp = loss_siamese(s_a1,s_a2)
    loss_m = loss_travel_temp + loss_siamese_temp
    print("Loss m: ", loss_m)
    #get gen and siam loss and bptt
    loss_g = g_loss_f(fake)
    print("Loss g: ", loss_g)
    lossgtot = loss_g+10.*loss_m #+0.5*loss_id #CHANGE LOSS WEIGHTS HERE  (COMMENT OUT +w*loss_id IF THE IDENTITY LOSS TERM IS NOT NEEDED)

    lossgtot.backward()
    optimG.step()
    
    #get critic loss and bptt
    optimD.zero_grad()

    loss_dr = d_loss_r(real)
    loss_df = d_loss_f(fake)
    loss_d = (loss_dr+loss_df)/2.

    print("Loss d: ", loss_d)
    loss_d.backward()
    optimD.step()
    
    return loss_dr,loss_df,loss_g,loss_d

def train_d(a,b):
    optimD.zero_grad()
    
    a1,a2,a3 = extract_image(a)
    
    #translating A to B
    g_a1 = netG.forward(a1)
    g_a2 = netG.forward(a2)
    g_a3 = netG.forward(a3)
    #concatenate/assemble converted spectrograms
    g_a_total = assemble_image([g_a1,g_a2,g_a3])

    #feed concatenated spectrograms to critic
    fake = critic.forward(g_a_total)
    real = critic.forward(b)

    #get critic loss and bptt
    loss_dr = d_loss_r(real)
    loss_df = d_loss_f(fake)
    loss_d = (loss_dr+loss_df)/2.
    # loss_d = torch.autograd.Variable(loss_d, requires_grad = True)
    
    loss_d.backward()
    optimD.step()

    return loss_dr,loss_df

def train(args):
    ## 트레이닝 파라메터 설정하기
    mode = args.mode
    gen_lr = args.g_lr
    dis_lr = args.d_lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    gupt = args.gupt  #only train discriminator every gupt'th batch
    train_continue = args.train_continue

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)

    print("G_learning rate: %.4e" % gen_lr)
    print("D_learning rate: %.4e" % dis_lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    ## 데이터 load
    jazz_gtzan = np.load(os.path.join(data_dir,'jazz_other_mel_gtzan.npy'))
    jazz_melon = np.load(os.path.join(data_dir,'jazz_other_mel_melon.npy'))
    rock_gtzan = np.load(os.path.join(data_dir,'rock_other_mel_gtzan.npy'))
    rock_melon = np.load(os.path.join(data_dir,'rock_other_mel_melon.npy'))

    jazz_all = np.concatenate((jazz_gtzan,jazz_melon))
    rock_all = np.concatenate((rock_gtzan,rock_melon))

    a_dataset = AudioDataset(jazz_all, hop=hop, shape=shape)
    a_loader = DataLoader(dataset=a_dataset,batch_size=batch_size,shuffle=True)

    b_dataset = AudioDataset(rock_all, hop=hop, shape=shape)
    b_loader = DataLoader(dataset=b_dataset,batch_size=batch_size,shuffle=True)

    ## 네트워크 생성
    netG = Generator((hop,shape,1)).to(device)
    netS = Siamese((hop,shape,1)).to(device)
    netD = Discriminator((hop,3*shape,1)).to(device)

    #Generator loss is a function of 
    params = list(netG.parameters()) + list(netS.parameters())

    optimG = optim.Adam(params, lr=gen_lr)
    optimD = optim.SGD(critic.parameters(), lr=dis_lr)


    st_epoch = 0

    if mode == 'train':
        if train_continue == "on": # 이어서 학습할때 on, 처음 학습 시킬때는 off
            netG,netD,netS, optimG,optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG,netD=netD, optimG=optimG,optimD=optimD)


        update_lr(gen_lr, dis_lr)
        df_list = []
        dr_list = []
        g_list = []
        id_list = []
        g = 0
    
        for epoch in range(num_epoch):
            for batchi,(a,b) in enumerate(zip(a_loader,b_loader)):

                if batchi%gupt==0:
                    dloss_t,dloss_f,gloss,idloss = train_all(a,b)
                else:
                    dloss_t,dloss_f = train_d(a,b)

                df_list.append(dloss_f)
                dr_list.append(dloss_t)
                g_list.append(gloss)
                id_list.append(idloss)
                g += 1

                if batchi % 100==0:
                    print('[Epoch : {}/{}] [Batch : {}] [D loss f: {}]'.format(epoch,num_epoch,batchi,(sum(df_list[-g:])/len(df_list[-g:]))), end=' ')
                    print('D_loss_T : {}]'.format((sum(dr_list[-g:])/len(dr_list[-g:]))), end=' ')
                    print('[G loss: {}]'.format((sum(g_list[-g:])/len(g_list[-g:]))), end=' ')
                    print('[ID loss: {}]'.format((sum(id_list[-g:])/len(id_list[-g:]))), end=' ')
                    g = 0

            if epoch % 100 == 0:
                save(ckpt_dir=ckpt_dir, netG=netG, netD = netD,netS=netS ,optimG=optimG, optimD=optimD, epoch=epoch)

    if mode == 'test':
        netG, netD,netS, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG,netS=netS ,netD = netD, optimG=optimG, optimD = optimD)

        test_data = np.load(os.path.join(data_dir,'testset.npy'))
        dataset_test = AudioDataset(test_data, hop=hop, shape=shape)
        loader_test = DataLoader(dataset=dataset_test,batch_size=batch_size,shuffle=True) 

        with torch.no_grad():
            netG.eval()

            Loss = []

            for batch, data in enumerate(loader_test,1):
                a1,a2,a3 = extract_image(a) 

                #translating A to B
                g_a1 = netG.forward(a1)
                g_a2 = netG.forward(a2)
                g_a3 = netG.forward(a3)

                g_a_total = assemble_image([g_a1,g_a2,g_a3])




