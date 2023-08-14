import torch
from torch import nn


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class AdaIN_SET(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, content, style_mean, style_std):
        assert style_mean is not None
        assert style_std is not None

        size = content.size()

        content_mean, content_std = calc_mean_std(content)

        style_mean = style_mean.reshape(size[0],content_mean.shape[1],1,1)
        style_std = style_std.reshape(size[0],content_mean.shape[1],1,1)
        
        normalized_feat = (content - content_mean.expand(size)) / content_std.expand(size)
        sum_mean = style_mean.expand(size)
        sum_std = style_std.expand(size)

        return normalized_feat*sum_std + sum_mean


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)


        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LB(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LB, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LB(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LB(dim, dim, norm=norm, activation=activ)]
        self.model += [LB(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x

class HierarchyFilter(nn.Module):
    def __init__(self,in_channel,out_channel,use_adain=True,gamma=0.5,fix_gamma=True):
        super(HierarchyFilter,self).__init__()
        self.feat = None
        self.out_channel = out_channel
        self.input_channel = in_channel
        self.use_adain = use_adain

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel*2, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.InstanceNorm2d(in_channel*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channel*2, out_channels=in_channel*2, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.InstanceNorm2d(in_channel*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=3, stride=1, padding=1, dilation=1)
        )
        self.gamma = nn.Parameter(torch.tensor(gamma))
        if fix_gamma:
            self.gamma.requires_grad = False
        else:
            self.gamma.requires_grad = True

        if self.use_adain:
            self.adain_set = AdaIN_SET()
            self.mlp = MLP(8, out_channel*2, out_channel*3, 3, norm='none', activ='relu')

    def forward(self, input):
        b_size, n_channel, height, width = input.shape

        feature = self.net(input)
        self.feat = feature

        out = input - feature[:,0:n_channel]
        tmp_out = out
        for i in range(n_channel,self.out_channel,n_channel):
            tmp_out = tmp_out - feature[:,i:i+n_channel]
            out = torch.cat((out,tmp_out),1)

        return out

    def reverse(self, input, style, rec=False):
        feature = self.feat

        if not rec and self.use_adain:
            pred_style = self.mlp(style)
            mean, std = pred_style.chunk(2, 1)
            input = self.adain_set(input, mean, std)

        out = input[:,-self.input_channel:] + feature[:,-self.input_channel:]
        num = self.out_channel//self.input_channel

        for i in range(2,num+1):

            out =  (out*self.gamma + input[:,-i*self.input_channel:-(i-1)*self.input_channel]*(1-self.gamma)) + feature[:,-i*self.input_channel:-(i-1)*self.input_channel] 

        return out


class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

