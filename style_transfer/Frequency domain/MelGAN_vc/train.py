from model import *
from dataset import *
from utils import *

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

    ## weights initialize
    init_weights(netG,init_type='normal',init_gain=0.02)
    init_weights(netS,init_type='normal',init_gain=0.02)
    init_weights(netD,init_type='normal',init_gain=0.02)

    ## optimizer 설정
    params = list(netG.parameters()) + list(netS.parameters())

    optimG = optim.Adam(params, lr=gen_lr, betas = (0.5,0.999))
    optimD = optim.Adam(netD.parameters(), lr=dis_lr,betas =(0.5,0.999))

    def update_lr(gen_lr, dis_lr):
        optimG.lr = gen_lr
        optimD.lr = dis_lr


    st_epoch = 0

    if mode == 'train':
        if train_continue == "on": # 이어서 학습할때 on, 처음 학습 시킬때는 off
            netG,netD,netS, optimG,optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG,netD=netD, optimG=optimG,optimD=optimD)


        update_lr(gen_lr, dis_lr)
        # d_list = []
        # g_list = []
        # g = 0
    
        for epoch in range(num_epoch):
            for batchi,(a,b) in enumerate(zip(a_loader,b_loader)):

                if batchi%gupt==0:
                    a1,a2,a3 = extract_image(a)    
                    b1,b2,b3 = extract_image(b)

                    optimG.zero_grad()

                    g_a1 = netG.forward(a1)
                    g_a2 = netG.forward(a2)
                    g_a3 = netG.forward(a3)

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
                    loss_s = loss_travel_temp + loss_siamese_temp
                    # print("Loss m: ", loss_m)
                    #get gen and siam loss and bptt
                    loss_g = g_loss_f(fake)
                    # print("Loss g: ", loss_g)
                    lossgtot = loss_g+10.*loss_s #+0.5*loss_id #CHANGE LOSS WEIGHTS HERE  (COMMENT OUT +w*loss_id IF THE IDENTITY LOSS TERM IS NOT NEEDED)

                    lossgtot.backward()
                    optimG.step()
                    
                    #get critic loss and bptt
                    optimD.zero_grad()

                    loss_dr = d_loss_r(real)
                    loss_df = d_loss_f(fake)
                    loss_d = (loss_dr+loss_df)/2.

                    # print("Loss d: ", loss_d)
                    loss_d.backward()
                    optimD.step()


                else:
                    optimD.zero_grad()
    
                    a1,a2,a3 = extract_image(a)
                    
                    #translating A to B
                    g_a1 = netG.forward(a1)
                    g_a2 = netG.forward(a2)
                    g_a3 = netG.forward(a3)
                    #concatenate/assemble converted spectrograms
                    g_a_total = assemble_image([g_a1,g_a2,g_a3])

                    #feed concatenated spectrograms to critic
                    fake = netD.forward(g_a_total)
                    real = netD.forward(b)

                    #get critic loss and bptt
                    loss_dr = d_loss_r(real)
                    loss_df = d_loss_f(fake)
                    loss_d = (loss_dr+loss_df)/2.
                    # loss_d = torch.autograd.Variable(loss_d, requires_grad = True)
                    
                    loss_d.backward()
                    optimD.step()

                # d_list.append(loss_d)
                # g_list.append(lossgtot)

            if epoch % 1 == 0:
                    print('[Epoch : {}/{}]'.format(epoch,num_epoch))
                    print('D_loss : {}]'.format(loss_d))
                    print('[G_loss: {}]'.format(lossgtot))
                    print('[S_loss: {}]'.format(loss_s))
                    

            if epoch % 100 == 0:
                save(ckpt_dir=ckpt_dir, netG=netG, netD = netD,netS=netS ,optimG=optimG, optimD=optimD, epoch=epoch)
                print('Save complet_{}_epoch'.format(epoch))
   
    if mode == 'test':
        netG, netD,netS, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG,netS=netS ,netD = netD, optimG=optimG, optimD = optimD)

        # gen = netG

        wv = np.load('Autumn Leaves_0_other.npy')                           #Load waveform
        sr = 22050
        print(wv.shape)
        spec = prep(wv)                                                     # Waveform to Spectrogram

        plt.figure(figsize=(50,1))                                          # Show Spectrogram
        plt.imshow(np.flip(spec, axis=0), cmap=None)
        plt.axis('off')
        plt.show()

        # abwv = towave(spec = spec, name='autumn_leavers_0_other',net=netG.cpu(), path='./result_jazz2rock/')           #Convert and save wav

        netG.cpu()
        specarr = chopspec(spec)
        print(specarr.shape)
        tem = specarr
        print('Generating...')
        a = torch.Tensor(tem).permute(0,3,1,2)
        ab = netG.forward(a)
        print('Assembling and Converting...')
        a = specass(a,spec)
        ab = ab.detach().numpy()
        ab = specass(ab,spec)
        awv = deprep(a)
        abwv = deprep(ab)
        print('Saving...')
        result_dir = './result_jazz2rock/autumn_leaves_0_other'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        sf.write(os.path.join(result_dir,'J2R.wav'), abwv, sr)
        sf.write(os.path.join(result_dir,'Jazz.wav'), awv, sr)
        print('Saved WAV!')
        IPython.display.display(IPython.display.Audio(np.squeeze(abwv), rate=sr))
        IPython.display.display(IPython.display.Audio(np.squeeze(awv), rate=sr))
        show = True
        if show:
            fig, axs = plt.subplots(ncols=2)
            axs[0].imshow(np.flip(a, -2), cmap=None)
            axs[0].axis('off')
            axs[0].set_title('Source')
            axs[1].imshow(np.flip(ab, -2), cmap=None)
            axs[1].axis('off')
            axs[1].set_title('Generated')
            plt.show()
        return abwv

        # abwv = towave(speca, name='autumn_leavers_0_other', path='./result_jazz2rock/')           #Convert and save wav




