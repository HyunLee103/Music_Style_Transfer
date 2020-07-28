import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import logging
from tqdm import tqdm
from collections import deque

###########################################################
# Encoder
class DilatedResConv(nn.Module):
    def __init__(self, channels, dilation=1, activation='relu', padding=1, kernel_size=3, left_pad=0):
        super().__init__()
        in_channels = channels

        if activation == 'relu':
            self.activation = torch.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'glu':
            self.activation = torch.glu
            in_channels = channels // 2

        self.left_pad = left_pad
        self.dilated_conv = nn.Conv1d(in_channels, channels, kernel_size=kernel_size, stride=1,
                                      padding=dilation * padding, dilation=dilation, bias=True)
        self.conv_1x1 = nn.Conv1d(in_channels, channels,
                                  kernel_size=1, bias=True)

    def forward(self, input):
        x = input

        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))
        x = self.dilated_conv(x)
        x = self.activation(x)
        x = self.conv_1x1(x)

        return input + x

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.n_blocks = args["encoder_blocks"]
        self.n_layers = args["encoder_layers"]
        self.channels = args["encoder_channels"]
        self.latent_channels = args["latent_d"]
        self.activation = args["encoder_func"]
        self.encoder_pool = 800

        layers = []
        for _ in range(self.n_blocks):
            for i in range(self.n_layers):
                dilation = 2 ** i
                layers.append(DilatedResConv(self.channels, dilation, self.activation))
        self.dilated_convs = nn.Sequential(*layers)

        self.start = nn.Conv1d(1, self.channels, kernel_size=3, stride=1,
                               padding=1)
        self.conv_1x1 = nn.Conv1d(self.channels, self.latent_channels, 1)
        self.pool = nn.AvgPool1d(self.encoder_pool)

    def forward(self, x):
        x = x / 255 - .5
        if x.dim() < 3:
            x = x.unsqueeze(1)

        x = self.start(x)
        x = self.dilated_convs(x)
        x = self.conv_1x1(x)
        x = self.pool(x)

        return x
    
########################################################
# Decoder
class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 dilation=1,
                 **kwargs):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=dilation * (kernel_size - 1),
            dilation=dilation,
            **kwargs)

    def forward(self, input):
        out = super(CausalConv1d, self).forward(input)
        return out[:, :, :-self.padding[0]]


class WavenetLayer(nn.Module):
    def __init__(self, residual_channels, skip_channels, cond_channels,
                 kernel_size=2, dilation=1):
        super(WavenetLayer, self).__init__()

        self.causal = CausalConv1d(residual_channels, 2 * residual_channels,
                                   kernel_size, dilation=dilation, bias=True)
        self.condition = nn.Conv1d(cond_channels, 2 * residual_channels,
                                   kernel_size=1, bias=True)
        self.residual = nn.Conv1d(residual_channels, residual_channels,
                                  kernel_size=1, bias=True)
        self.skip = nn.Conv1d(residual_channels, skip_channels,
                              kernel_size=1, bias=True)

    def _condition(self, x, c, f):
        c = f(c)
        x = x + c
        return x

    def forward(self, x, c=None):
        x = self.causal(x)
        if c is not None:
            x = self._condition(x, c, self.condition)

        assert x.size(1) % 2 == 0
        gate, output = x.chunk(2, 1)
        gate = torch.sigmoid(gate)
        output = torch.tanh(output)
        x = gate * output

        residual = self.residual(x)
        skip = self.skip(x)

        return residual, skip


class WaveNet(nn.Module):
    def __init__(self, args, create_layers=True, shift_input=True):
        super().__init__()

        self.blocks = args["blocks"]
        self.layer_num = args["layers"]
        self.kernel_size = args["kernel_size"]
        self.skip_channels = args["skip_channels"]
        self.residual_channels = args["residual_channels"]
        self.cond_channels = args["latent_d"]
        self.classes = 256
        self.shift_input = shift_input

        if create_layers:
            layers = []
            for _ in range(self.blocks):
                for i in range(self.layer_num):
                    dilation = 2 ** i
                    layers.append(WavenetLayer(self.residual_channels, self.skip_channels, self.cond_channels,
                                               self.kernel_size, dilation))
            self.layers = nn.ModuleList(layers)

        self.first_conv = CausalConv1d(1, self.residual_channels, kernel_size=self.kernel_size)
        self.skip_conv = nn.Conv1d(self.residual_channels, self.skip_channels, kernel_size=1)
        self.condition = nn.Conv1d(self.cond_channels, self.skip_channels, kernel_size=1)
        self.fc = nn.Conv1d(self.skip_channels, self.skip_channels, kernel_size=1)
        self.logits = nn.Conv1d(self.skip_channels, self.classes, kernel_size=1)

    def _condition(self, x, c, f):
        c = f(c)
        x = x + c
        return x

    @staticmethod
    def _upsample_cond(x, c):
        bsz, channels, length = x.size()
        cond_bsz, cond_channels, cond_length = c.size()
        assert bsz == cond_bsz

        if c.size(2) != 1:
            c = c.unsqueeze(3).repeat(1, 1, 1, length // cond_length)
            c = c.view(bsz, cond_channels, length)

        return c

    @staticmethod
    def shift_right(x):
        x = F.pad(x, (1, 0))
        return x[:, :, :-1]

    def forward(self, x, c=None):
        if x.dim() < 3:
            x = x.unsqueeze(1)
        if (not 'Half' in x.type()) and (not 'Float' in x.type()):
            x = x.float()

        x = x / 255 - 0.5

        if self.shift_input:
            x = self.shift_right(x)

        if c is not None:
            c = self._upsample_cond(x, c)

        residual = self.first_conv(x)
        skip = self.skip_conv(residual)

        for layer in self.layers:
            r, s = layer(residual, c)
            residual = residual + r
            skip = skip + s

        skip = torch.relu(skip)
        skip = self.fc(skip)
        if c is not None:
            skip = self._condition(skip, c, self.condition)
        skip = torch.relu(skip)
        skip = self.logits(skip)

        return skip

    ### Weights ###
    def export_layer_weights(self):
        Wdilated, Bdilated = [], []
        Wres, Bres = [], []
        Wskip, Bskip = [], []

        for l in self.layers:
            Wdilated.append(l.causal.weight)
            Bdilated.append(l.causal.bias)

            Wres.append(l.residual.weight)
            Bres.append(l.residual.bias)

            Wskip.append(l.skip.weight)
            Bskip.append(l.skip.bias)

        return Wdilated, Bdilated, Wres, Bres, Wskip, Bskip

    def export_embed_weights(self):
        inp = torch.range(0, 255) / 255 - 0.5
        prev = self.first_conv.weight[:, :, 0].cpu().contiguous()
        prev = inp.unsqueeze(1) @ prev.transpose(0, 1)
        prev = prev + self.first_conv.bias.cpu() / 2

        curr = self.first_conv.weight[:, :, 1].cpu().contiguous()
        curr = inp.unsqueeze(1) @ curr.transpose(0, 1)
        curr = curr + self.first_conv.bias.cpu() / 2

        return prev, curr

    def export_final_weights(self):
        Wzi = self.skip_conv.weight
        Bzi = self.skip_conv.bias
        Wzs = self.fc.weight
        Bzs = self.fc.bias
        Wza = self.logits.weight
        Bza = self.logits.bias

        return Wzi, Bzi, Wzs, Bzs, Wza, Bza
    
################################################################
# Discriminator

class ZDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_classes = args["n_datasets"]
        self.d_channels = args["d_channels"]
        self.dropout_rate = args["dropout_rate"]

        convs = []
        for i in range(args["d_layers"]):
            in_channels = args["latent_d"] if i == 0 else self.d_channels
            convs.append(nn.Conv1d(in_channels, self.d_channels, 1))
            convs.append(nn.ELU())
        convs.append(nn.Conv1d(self.d_channels, self.n_classes, 1))

        self.convs = nn.Sequential(*convs)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, z):
        z = self.dropout(z)
        logits = self.convs(z)  # (N, n_classes, L)

        mean = logits.mean(2)
        return mean
    
##################################################################
# Generator
class QueuedConv1d(nn.Module):
    def __init__(self, conv, data):
        super().__init__()
        if isinstance(conv, nn.Conv1d):
            self.inner_conv = nn.Conv1d(conv.in_channels,
                                        conv.out_channels,
                                        conv.kernel_size)
            self.init_len = conv.padding[0]
            self.inner_conv.weight.data.copy_(conv.weight.data)
            self.inner_conv.bias.data.copy_(conv.bias.data)

        elif isinstance(conv, QueuedConv1d):
            self.inner_conv = nn.Conv1d(conv.inner_conv.in_channels,
                                        conv.inner_conv.out_channels,
                                        conv.inner_conv.kernel_size)
            self.init_len = conv.init_len
            self.inner_conv.weight.data.copy_(conv.inner_conv.weight.data)
            self.inner_conv.bias.data.copy_(conv.inner_conv.bias.data)

        self.init_queue(data)

    def init_queue(self, data):
        self.queue = deque([data[:, :, 0:1]]*self.init_len,
                           maxlen=self.init_len)

    def forward(self, x):
        y = x
        x = torch.cat([self.queue[0], x], dim=2)
        # import pdb; pdb.set_trace()
        self.queue.append(y)

        return self.inner_conv(x)


class WavenetGenerator(nn.Module):
    Q_ZERO = 128

    def __init__(self, wavenet: WaveNet, batch_size=1, cond_repeat=800, wav_freq=16000):
        super().__init__()
        self.wavenet = wavenet
        self.wavenet.shift_input = False
        self.cond_repeat = cond_repeat
        self.wav_freq = wav_freq
        self.batch_size = batch_size
        self.was_cuda = next(self.wavenet.parameters()).is_cuda

        x = torch.zeros(self.batch_size, 1, 1)
        x = x.cuda() if self.was_cuda else x
        self.wavenet.first_conv = QueuedConv1d(self.wavenet.first_conv, x)

        x = torch.zeros(self.batch_size, self.wavenet.residual_channels, 1)
        x = x.cuda() if self.was_cuda else x
        for layer in self.wavenet.layers:
            layer.causal = QueuedConv1d(layer.causal, x)

        if self.was_cuda:
            self.wavenet.cuda()
        self.wavenet.eval()

    def forward(self, x, c=None):
        return self.wavenet(x, c)

    def reset(self):
        return self.init()

    def init(self, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size

        x = torch.zeros(self.batch_size, 1, 1)
        x = x.cuda() if self.was_cuda else x
        self.wavenet.first_conv.init_queue(x)

        x = torch.zeros(self.batch_size, self.wavenet.residual_channels, 1)
        x = x.cuda() if self.was_cuda else x
        for layer in self.wavenet.layers:
            layer.causal.init_queue(x)

        if self.was_cuda:
            self.wavenet.cuda()

    @staticmethod
    def softmax_and_sample(prediction, method='sample'):
        if method == 'sample':
            probabilities = F.softmax(prediction)
            samples = torch.multinomial(probabilities, 1)
        elif method == 'max':
            _, samples = torch.max(F.softmax(prediction), dim=1)
        else:
            assert False, "Method not supported."

        return samples

    def generate(self, encodings, init=True, method='sample'):
        if init:
            self.init(encodings.size(0))

        samples = torch.zeros(encodings.size(0), 1, encodings.size(2)*self.cond_repeat + 1)
        samples.fill_(self.Q_ZERO)
        samples = samples.long()
        samples = samples.cuda() if encodings.is_cuda else samples

        with torch.no_grad():
            t0 = time.time()
            for t1 in tqdm(range(encodings.size(2)), desc='Generating'):
                for t2 in range(self.cond_repeat):
                    t = t1 * self.cond_repeat + t2
                    x = samples[:, :, t:t + 1].clone()
                    c = encodings[:, :, t1:t1+1]

                    prediction = self(x, c)[:, :, 0]

                    argmax = self.softmax_and_sample(prediction, method)

                    samples[:, :, t+1] = argmax

            logging.info(f'{encodings.size(0)} samples of {encodings.size(2)*self.cond_repeat/self.wav_freq} seconds length '
                         f'generated in {time.time() - t0} seconds.')

        return samples[:, :, 1:]
    