import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks
from .gaitgl import GeMHPP

class PermuteLayer(nn.Module):

    def __init__(self, *dims):
        """Applies transposition on a given Tensor given the axis
            specified
        """
        super(PermuteLayer, self).__init__()

        # Retrieve the dims to transpose
        self.dims = dims
        #self.ori_dims = list(range(dims))

    def forward(self, x):
        return PermuteLayer.permute(x, *self.dims)
    
    @staticmethod
    def permute(x, *dims):
        cur_per = list(range(len(dims)))     # Current dim order     idx: dim
        cur_dim_idx = list(range(len(dims))) # Current dim position  dim: idx

        # Transpose each dim pair until complete permutation
        for i in range(len(dims)):
        
            if dims[i] == cur_per[i]:
                # Transpose not required
                continue

            dst_dim_pos = i
            org_dim_pos = cur_dim_idx[dims[i]] #cur_per.index(dims[i]) 

            org_dim = cur_per[i]

            # Transpose dims
            x = torch.transpose(x, org_dim_pos, dst_dim_pos)

            cur_per[org_dim_pos], cur_per[dst_dim_pos] = cur_per[dst_dim_pos], cur_per[org_dim_pos]
            cur_dim_idx[org_dim], cur_dim_idx[dims[i]] =  cur_dim_idx[dims[i]], cur_dim_idx[org_dim]

        return x

class GLConv(nn.Module):

    """
    17/10/2022: Replaced permute layer operation by transpose layer.
    """
    def __init__(self, in_channels, out_channels, halving, fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, act_fn=F.leaky_relu,
                    layer_norm=False, batch_size=None, height=None, width=None, seq_length=None, **kwargs):
        super(GLConv, self).__init__()
        self.halving = halving
        self.fm_sign = fm_sign
        self.global_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
        self.local_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs) if halving >= 0 else None
        self.act_fn = act_fn    #(**kwargs)

        if layer_norm:
            """
            if batch_size is None or height is None or width is None or seq_length is None: #any((x not in kwargs) for x in ('batch_size', 'height', 'width', 'seq_length')):
                raise ValueError('To enable layer normalization, following paramteters must be provided: batch_size, height, width and seq_length')
            inp_layer_norm_size = self.get_out_shape(batch_size=batch_size, height=height, width=width, seq_length=seq_length)
            self.layer_norm_lay = nn.LayerNorm(inp_layer_norm_size[1:])
            """
            #norm_channels = out_channels if halving < 0 or not fm_sign else 2 * out_channels
            if layer_norm == 2: # True or 1: LayerChannelNorm, 2: BatchNorm
                self.layer_norm_lay = nn.BatchNorm3d(out_channels)
            else:
                self.layer_norm_lay = LayerChannelNorm(normalized_shape=out_channels, channel_dim=1, num_dims=5)
        else:
            self.layer_norm_lay = None

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        gob_feat = self.global_conv3d(x)
        if self.halving == 0:
            lcl_feat = self.local_conv3d(x)
        elif self.halving > 0:
            h = x.size(3)
            split_size = int(h // 2**self.halving)
            lcl_feat = x.split(split_size, 3)
            lcl_feat = torch.cat([self.local_conv3d(_) for _ in lcl_feat], 3)

        if self.halving < 0:
            if self.layer_norm_lay:
                # Normalize input
                gob_feat = self.layer_norm_lay(gob_feat)

                feat = self.act_fn(gob_feat)
                #feat = self.act_fn(self.layer_norm_lay(gob_feat))
            else:
                feat = self.act_fn(gob_feat)
        elif not self.fm_sign:
            if self.layer_norm_lay:
                # Normalize input
                gob_feat = self.layer_norm_lay(gob_feat)
                lcl_feat = self.layer_norm_lay(lcl_feat)

                feat = self.act_fn(gob_feat) + self.act_fn(lcl_feat)
            else:
                feat = self.act_fn(gob_feat) + self.act_fn(lcl_feat)
        else:
            if self.layer_norm_lay:
                cat_feat = torch.cat([gob_feat, lcl_feat], dim=3)
                # Normalize input
                cat_feat = self.layer_norm_lay(cat_feat)

                feat = self.act_fn(cat_feat)
            else:
                feat = self.act_fn(torch.cat([gob_feat, lcl_feat], dim=3))
        return feat

    def get_out_shape(self, batch_size, height, width, seq_length):
	    num_halves = int(height // 2**self.halving) if self.halving > 0 else (1 if self.halving == 0 else 0) # ((num_halves if self.fm_sign else 0) + 1)

	    batch_size, out_shape, seq_length, height, width = self.global_conv3d.get_out_shape(batch_size, height, width, seq_length)
	    return batch_size, out_shape, seq_length, (2 if self.fm_sign else 1) * height, width


class LayerChannelNorm(nn.Module):
    """
       Reimplemented from torch.nn.LayerNorm.
       It allows to apply layer normalization specifically to the channel dim
    """
    def __init__(self, normalized_shape, channel_dim, num_dims, eps=1e-05, elementwise_affine=True, device=None, dtype=None, **kwargs):
        super(LayerChannelNorm, self).__init__()
        
        # Private attributes
        self.normalized_shape = normalized_shape
        self.channel_dim = channel_dim
        self.num_dims = num_dims
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.device = device
        self.dtype = dtype
        self.kwargs = kwargs
        
        # Normal layer norm
        self.layer_norm = nn.LayerNorm(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype, **kwargs)
    
    def forward(self, x):
    
        dim0, dim1 = self.channel_dim, self.num_dims - 1
   
        # Swap channel dim to last dim and normalize over it
        if dim0 < dim1:
            x1 = x.transpose(dim0, dim1)
            x1 = self.layer_norm(x1)
            x1 = x1.transpose(dim1, dim0)
        else:
            x1 = self.layer_norm(x)
        
        return x1
    
    def __repr__(self):
        return f'LayerChannelNorm(normalized_shape={self.normalized_shape}, channel_dim={self.channel_dim}, num_dims={self.num_dims}, eps={self.eps}, elementwise_affine={self.elementwise_affine}, device={self.device}, dtype={self.dtype}{(", " if self.kwargs else "") + ", ".join(str(k)+"="+str(v) for k, v in self.kwargs.items())})'

def apply_rep_layers(layers, data, sel_index=None, **kwargs):
    out = data
    for lay in layers:
        out = lay(out, **kwargs)
        if sel_index:
            out = out[sel_index]
    
    return out

class GaitGL_Multichannel4(BaseModel):
    """
        Extended version of GaitGL to support multiple channels at input.
        In addition, following features are supported:
        -   Repeated layers are supported by the key repeat_layers
        -   Subset of the original frame channels can be selected for training by using the key channels_in_use
        -   Layer Norm batch is supported on each layer

        GaitGL: Gait Recognition via Effective Global-Local Feature Representation and Local Temporal Aggregation
        Arxiv : https://arxiv.org/pdf/2011.01461.pdf
        
        
        Updated: 30/04/2024
    """

    def __init__(self, *args, **kargs):
        super(GaitGL_Multichannel4, self).__init__(*args, **kargs)
        # Private attributes
        self._in_channels = None

    @property
    def in_channels(self):
        return self._in_channels

    def build_network(self, model_cfg):
        in_channel = model_cfg['num_in_channels']
        in_c = model_cfg['channels']
        class_num = model_cfg['class_num']
        dataset_name = self.cfgs['data_cfg']['dataset_name']
        self.channels_in_use = self.cfgs['data_cfg'].pop('channels_in_use', list(range(in_channel)))

        # Adjust the actual number of input channels that will be used
        in_channel = len(self.channels_in_use)

        # Update attributes
        self._in_channels = in_channel

        if dataset_name in ['OUMVLP','GREW']:
            halving = model_cfg.pop('halving', [1, 1, 1, 1, 1, 1])
            layer_norm = model_cfg.pop('layer_norm', [0, 0, 0, 0, 0, 0])

            # For OUMVLP and GREW
            self.conv3d = nn.Sequential(
                BasicConv3d(in_channel, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
                BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
            )
            self.LTA = nn.Sequential(
                BasicConv3d(in_c[0], in_c[0], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(inplace=True)
            )

            self.GLConvA0 = nn.Sequential(
                GLConv(in_c[0], in_c[1], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[1], in_c[1], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
            self.MaxPool0 = nn.MaxPool3d(
                kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.GLConvA1 = nn.Sequential(
                GLConv(in_c[1], in_c[2], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[2], in_c[2], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
            self.GLConvB2 = nn.Sequential(
                GLConv(in_c[2], in_c[3], halving=1, fm_sign=False,  kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[3], in_c[3], halving=1, fm_sign=True,  kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
        else:
            halving = model_cfg.pop('halving', [3, 3, 3])
            layer_norm = model_cfg.pop('layer_norm', [0, 0, 0, 0, 0, 0])
            rep_layers = model_cfg.get('repeat_layers', [1,] * 12)

            # For CASIA-B or other unstated datasets.
            ##  Input Conv3D layer (For each layer, output dimension needs to be computed)
            self.conv3d = []
            self.out_shape = []

            self.conv3d.append(BasicConv3d(in_channel, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)))

            if layer_norm[0]:
                self.conv3d.append(LayerChannelNorm(normalized_shape=in_c[0], channel_dim=1, num_dims=5))

            self.conv3d.append(nn.LeakyReLU(inplace=True))
            
            for _ in range(1, rep_layers[0]): 
                self.conv3d.append(BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                                    stride=(1, 1, 1), padding=(1, 1, 1)))

                if layer_norm[0]:
                    self.conv3d.append(LayerChannelNorm(normalized_shape=in_c[0], channel_dim=1, num_dims=5))

                self.conv3d.append(nn.LeakyReLU(inplace=True))

            self.conv3d = nn.Sequential(*self.conv3d)

            ## LTA layer
            self.LTA = []
            for _ in range(rep_layers[1]):
                self.LTA.append(BasicConv3d(in_c[0], in_c[0], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)))

                if layer_norm[1]:
                    self.LTA.append(LayerChannelNorm(normalized_shape=in_c[0], channel_dim=1, num_dims=5))

                self.LTA.append(nn.LeakyReLU(inplace=True))

            self.LTA = nn.Sequential(*self.LTA)

            ## GLConvA0 layer
            self.GLConvA0 = []
            for _ in range(rep_layers[2]):
                self.GLConvA0.append(GLConv(in_c[0], in_c[1], halving=halving[0], fm_sign=False, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), layer_norm=layer_norm[2]))

            self.GLConvA0 = nn.Sequential(*self.GLConvA0)
            
            ## MaxPool0
            self.MaxPool0 = []
            for _ in range(rep_layers[3]):
                self.MaxPool0.append(nn.MaxPool3d(
                kernel_size=(1, 2, 2), stride=(1, 2, 2)))

                if layer_norm[3]:
                    self.MaxPool0.append(LayerChannelNorm(normalized_shape=in_c[1], channel_dim=1, num_dims=5))

            self.MaxPool0 = nn.Sequential(*self.MaxPool0)
            
            ## GLConvA1
            self.GLConvA1 = []
            for _ in range(rep_layers[4]):
                self.GLConvA1.append(GLConv(in_c[1], in_c[2], halving=halving[1], fm_sign=False, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), layer_norm=layer_norm[4]))

            self.GLConvA1 = nn.Sequential(*self.GLConvA1)

            ## GLConvB2
            self.GLConvB2 = []
            for _ in range(rep_layers[5]):
                self.GLConvB2.append(GLConv(in_c[2], in_c[2], halving=halving[2], fm_sign=True,  kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), layer_norm=layer_norm[5]))

            self.GLConvB2 = nn.Sequential(*self.GLConvB2)


        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = nn.Sequential(*[GeMHPP() for _ in range(rep_layers[7])])
        
        self.Head0 = nn.Sequential(*[SeparateFCs(64, in_c[-1], in_c[-1]) for _ in range(rep_layers[8])])
        
        if 'SeparateBNNecks' in model_cfg.keys():
            self.BNNecks = nn.Sequential(*[SeparateBNNecks(**model_cfg['SeparateBNNecks']) for _ in range(rep_layers[9])])
            self.Bn_head = False
        else:
            self.Bn = nn.Sequential(*[nn.BatchNorm1d(in_c[-1]) for _ in range(rep_layers[10])])
            self.Head1 = nn.Sequential(*[SeparateFCs(64, in_c[-1], class_num) for _ in range(rep_layers[11])])
            self.Bn_head = True

        #summary(self, (None, in_channel, 30, 64, 44))

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        
        # Isolate the selected index
        if len(ipts[0].size()) == 4:
            sils_prev = ipts[0].unsqueeze(2)
        else:
            sils_prev = ipts[0]

        sils_prev = torch.index_select(sils_prev, 2, torch.tensor(self.channels_in_use, device=sils_prev.device))
        sils = sils_prev.transpose(1, 2).contiguous() # [n, j, s, h, w] from [n, s, j, h, w]
        
        del ipts
        n, j, s, h, w = sils.size()
        n_prev, j_prev, s_prev, h_prev, w_prev = n, j, s, h, w

        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)

        outs = self.conv3d(sils)
        outs = self.LTA(outs)

        outs = self.GLConvA0(outs)
        outs = self.MaxPool0(outs)

        outs = self.GLConvA1(outs)
        outs = self.GLConvB2(outs)  # [n, c, s, h, w]

        outs = self.TP(outs, seqL=seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs = self.HPP(outs)  # [n, c, p]

        gait = self.Head0(outs)  # [n, c, p]
        
        if self.Bn_head: # Original GaitGL Head
            bnft = self.Bn(gait)  # [n, c, p]
            logi = self.Head1(bnft)  # [n, c, p]
            embed = bnft
        else: # BNNechk as Head
            bnft, logi = self.BNNecks(gait)  # [n, c, p]   
            embed = gait 

        n, j, s, h, w = sils.size()

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils_prev.view(n_prev * s_prev, j_prev, h_prev, w_prev) if j <= 3 else  sils_prev.view(n_prev * s_prev, j_prev, h_prev, w_prev)[:, :3, :, :]
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
