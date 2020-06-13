import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

"""
# mu-law function in 'music-translation' git
# Is it same with librosa.core.mu_compress()?
def mu_law(x, mu=255):
    x = numpy.clip(x, -1, 1)
    x_mu = numpy.sign(x) * numpy.log(1 + mu*numpy.abs(x))/numpy.log(1 + mu)
    return ((x_mu + 1)/2 * mu).astype('int16')


def inv_mu_law(x, mu=255.0):
    x = numpy.array(x).astype(numpy.float32)
    y = 2. * (x - (mu+1.)/2.) / (mu+1.)
    return numpy.sign(y) * (1./mu) * ((1. + mu)**numpy.abs(y) - 1.)
"""

"""
--CausalConv1d
--WavenetLayer
--WaveNet
remove condition func. We don't use condition
"""

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

        self.blocks = args.blocks
        self.layer_num = args.layers
        self.kernel_size = args.kernel_size
        self.skip_channels = args.skip_channels
        self.residual_channels = args.residual_channels
        self.cond_channels = args.latent_d
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
    def shift_right(x):  # hmm... help me hyun...
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

        skip = F.relu(skip)
        skip = self.fc(skip)
        if c is not None:
            skip = self._condition(skip, c, self.condition)
        skip = F.relu(skip)
        skip = self.logits(skip)

        return skip

"""
--ZDiscriminator--
args.d_channels == 100
args.latent_d == 128
"""
class ZDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_classes = 2

        convs = []
        for i in range(args.d_layers):  # d_layesrs default 3
            in_channels = args.latent_d if i == 0 else args.d_channels
            convs.append(nn.Conv1d(in_channels, args.d_channels, 1))
            convs.append(nn.ELU())
        convs.append(nn.Conv1d(args.d_channels, self.n_classes, 1))

        self.convs = nn.Sequential(*convs)
        self.dropout = nn.Dropout(p=args.p_dropout_discriminator)

    def forward(self, z):
        z = self.dropout(z)
        logits = self.convs(z)  # (N, n_classes, L)

        mean = logits.mean(2)
        return mean