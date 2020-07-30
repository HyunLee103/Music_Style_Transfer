from model import *
from dataset import *
from utils import *
import torch

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
    test_path = args.test_path

    only_D = args.only_D  #only train discriminator every gupt'th batch
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

    # dataloader
    if mode =='train':
        dataset = AudioDataset(data_dir,mode='train', direction='R2J')
        data_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

        ## 네트워크 생성
        netG = Generator((128,128,1)).to(device)
        netS = Siamese((128,128,1)).to(device)
        netD = Discriminator((128,3*128,1)).to(device)

        ## weights initialize
        init_weights(netG,init_type='normal',init_gain=0.02)
        init_weights(netS,init_type='normal',init_gain=0.02)
        init_weights(netD,init_type='normal',init_gain=0.02)

        ## Loss 설정
        BCE = torch.nn.BCELoss().to(device)
        BCE_logit = torch.nn.BCEWithLogitsLoss().to(device)
        L2 = torch.nn.MSELoss().to(device)


        ## optimizer 설정
        params_G = list(netG.parameters()) + list(netS.parameters())
        params_D = netD.parameters()

        optimG = torch.optim.Adam(params_G, lr=gen_lr, betas = (0.5,0.999))
        optimD = torch.optim.Adam(params_D, lr=dis_lr,betas =(0.5,0.999))

        # lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimG, lr_lambda=LambdaLR(num_epoch, 0, decay_epoch).step)
        # lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimD, lr_lambda=LambdaLR(num_epoch, 0, decay_epoch).step)

        st_epoch = 0

    
        if train_continue == "on": # 이어서 학습할때 on, 처음 학습 시킬때는 off
            netG,netD,netS, optimG,optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG,netD=netD,netS=netS, optimG=optimG,optimD=optimD)
        print('start training..')
        for epoch in range(st_epoch, num_epoch):
            netG.train()
            netD.train()
            netS.train()
            for batchi,data in enumerate(data_loader,1):
                # data load
                a = data['dataA'].to(device,dtype=torch.float)
                b = data['dataB'].to(device,dtype=torch.float)

                a1,a2,a3 = extract_image(a)
                b1,b2,b3 = extract_image(b)
             
                if batchi % only_D ==0:
                
                ## Generator

                    g_a1 = netG.forward(a1)
                    g_a2 = netG.forward(a2)
                    g_a3 = netG.forward(a3)

                    g_a_total = assemble_image([g_a1,g_a2,g_a3])

                    set_requires_grad(netD,True)
                    optimD.zero_grad()

                    real = netD.forward(b)
                    fake = netD.forward(g_a_total.detach())

                    loss_D_real = BCE_logit(real,torch.ones_like(real))
                    loss_D_fake = BCE_logit(fake,torch.zeros_like(fake))
                    loss_D = 0.5 * (loss_D_real+loss_D_fake)

                    loss_D.backward()
                    optimD.step()

                    set_requires_grad(netD,False)
                    optimG.zero_grad()
                
                    s_ga1 = netS.forward(g_a1)
                    s_ga2 = netS.forward(g_a3)
                    s_a1 = netS.forward(a1)
                    s_a2 = netS.forward(a3)


                    loss_S = loss_travel(s_a1,s_ga1,s_a2,s_ga2)
  

                    # s_1 = (s_a1 - s_a2)
                    # s_2 = (s_ga1 - s_ga2)
                    # # print(s_1.shape,s_2.shape)

                    # loss_S = L2(s_1, s_2)

                    real = netD.forward(b)
                    fake = netD.forward(g_a_total)

                    loss_G = BCE_logit(fake,torch.ones_like(fake))
                    # loss_G_real = BCE_logit(real,torch.zeros_like(fake))

                    loss_tot = loss_G +10.*loss_S

                    loss_tot.backward()
                    optimG.step()


                else:
                    g_a1 = netG.forward(a1)
                    g_a2 = netG.forward(a2)
                    g_a3 = netG.forward(a3)

                    g_a_total = assemble_image([g_a1,g_a2,g_a3])

                    fake = netD.forward(g_a_total)
                    real = netD.forward(b)

                    loss_dr = d_loss_r(real)
                    loss_df = d_loss_f(fake.detach())

                    loss_D = (loss_dr + loss_df) * 0.5

                    optimD.zero_grad()
                    loss_D.backward()
                    optimD.step()


            if epoch % 1 == 0:
                    print('[Epoch : {}/{}]'.format(epoch,num_epoch))
                    print('[D_loss : {}]'.format(loss_D))
                    print('[G_loss: {}]'.format(loss_G))
                    print('[S_loss: {}]'.format(loss_S))
                    

            if epoch % 100 == 0:
                save(ckpt_dir=ckpt_dir, netG=netG, netD = netD,netS=netS ,optimG=optimG, optimD=optimD, epoch=epoch)
                print('Save complet_{}_epoch'.format(epoch))
   
    if mode == 'test':
        print('start test..')
        # load test data
        data_test = AudioDataset(data_dir,mode='test', direction='R2J')
        test_loader = DataLoader(dataset=data_test,batch_size=1,shuffle=False)
        
        num_test = len(data_test)

        # load network
        netG = Generator((128,128,1)).to(device)


        # ## weights initialize
        # init_weights(netG,init_type='normal',init_gain=0.02)
        # init_weights(netS,init_type='normal',init_gain=0.02)
        # init_weights(netD,init_type='normal',init_gain=0.02)

        netG,epoch = load_test(ckpt_dir=ckpt_dir, netG=netG)

        # test
        with torch.no_grad():
            netG.eval()
            print('start inference..{}_epoch'.format(epoch))
            for i, data in enumerate(test_loader,1):

                a = data['dataA'].to(device,dtype=torch.float)
                # print(a.shape)
                a1,a2,a3 = extract_image(a)

                g_a1 = netG.forward(a1)
                g_a2 = netG.forward(a2)
                g_a3 = netG.forward(a3)
                print(g_a1.shape)

                g_a_total = assemble_image([g_a1,g_a2,g_a3])
                # origin_total = assemble_image([a1,a2,a3])

                g_a_total = g_a_total.cpu().numpy().reshape(-1,)
                # origin_total = origin_total.cpu().numpy().reshape(-1,)

                # print(g_a_total.shape,origin_total.shape)
                    

                try:
                    result_trans = np.concatenate((g_a_total,result_trans),axis=0)
                    # result_real = np.concatenate((origin_total,result_real),axis=0)

                except:
                    result_trans = g_a_total
                    # result_real = origin_total


            # inputs = np.load(os.path.join(data_dir,'wing_30.npy')) # put inference full audio 

            result_trans = result_trans.flatten()
            # result_real = result_real.flatten()

            # print(result_real.shape,result_trans.shape)
                                
            # librosa.output.write_wav('./result/input_{}.wav'.format(epoch),result_real,16384)
            librosa.output.write_wav('./result/output_{}.wav'.format(epoch),result_trans,16384)