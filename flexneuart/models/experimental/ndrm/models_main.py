#
# A slightly modified code (trying to change it as little as possible) from the following repo:
# https://github.com/bmitra-msft/TREC-Deep-Learning-Quick-Start/
# which is distributed under the Apache2-compatible MIT license
#

import torch
import torch.nn as nn
import numpy as np
from .conformer import PositionalEncoding, ConformerEncoderLayer, ConformerEncoder

#
# *IMPORTANT* REMINDER:
# Don't store the ref to the parent module here: this makes __repr__ function recurse infinitely!!!
# That's why NDRM[123] functions receive a ref to the wrapper module arguments, not to the module itself!
#
class NDRM1(nn.Module):

    def __init__(self, args):
        super(NDRM1, self).__init__()

        self.args = args
        self.embed = nn.Embedding(self.args.vocab_size, self.args.num_hidden_nodes)
        self.embed.weight = nn.Parameter(args.pretrained_embeddings, requires_grad=True)
        self.pos_encoder = PositionalEncoding(self.args.num_hidden_nodes, dropout=self.args.dropout, max_len=self.args.max_terms_doc)
        self.fc_qt = nn.Linear(self.args.num_hidden_nodes, self.args.num_hidden_nodes)
        enable_conformer = (not self.args.no_conformer)
        encoder_layers = ConformerEncoderLayer(self.args.num_hidden_nodes, self.args.num_attn_heads, self.args.num_hidden_nodes, self.args.dropout,
                                               attn=True, conv=enable_conformer, convsz=self.args.conv_window_size, shared_qk=enable_conformer, sep=enable_conformer)
        self.contextualize = ConformerEncoder(encoder_layers, self.args.num_encoder_layers)
        self.fc_ctx = nn.Linear(2, 1)
        self.cosine_sim = nn.CosineSimilarity(dim=-1)
        self.rbf_kernel = RBFKernel(args)

    def forward(self, q, d, mask_q, mask_d, qti_mode=False):
        q = self.embed(q)                                                           # [Q x Tq] --> [Q x Tq x H]
        q = self.fc_qt(q)                                                           # [Q x Tq x H] --> [Q x Tq x H]
        q = q.unsqueeze(dim=1)                                                      # [Q x Tq x H] --> [Q x 1 x Tq x H]
        q = q.unsqueeze(dim=-2)                                                     # [Q x 1 x Tq x H] --> [Q x 1 x Tq x 1 x H]
        d = self.embed(d)                                                           # [Q x D x Td] --> [Q x D x Td x H]
        shape_mask = mask_d.shape
        mask_d = mask_d.view(-1, shape_mask[-1])                                    # [Q x D x Td] --> [QD x Td]
        shape_d = d.shape
        d_ctx = d.view(-1, shape_d[2], shape_d[3])                                  # [Q x D x Td x H] --> [QD x Td x H]
        d_ctx = d_ctx.permute(1, 0, 2)                                              # [Q x D x Td x H] --> [Td x QD x H]
        d_ctx = self.pos_encoder(d_ctx)                                             # [Td x QD x H] --> [Td x QD x H]
        d_ctx = self.contextualize(d_ctx, src_key_padding_mask=~mask_d.bool())      # [Td x QD x H], [Q x D x Td] --> [Td x QD x H]
        d_ctx = d_ctx.permute(1, 0, 2)                                              # [Td x QD x H] --> [QD x Td x H]
        d_ctx = d_ctx.view(shape_d)                                                 # [QD x Td x H] --> [Q x D x Td x H]
        mask_d = mask_d.view(shape_mask)                                            # [QD x Td] --> [Q x D x Td]
        d = torch.stack([d, d_ctx], dim=-1)                                         # [Q x D x Td x H], [Q x D x Td x H] --> [Q x D x Td x H x 2]
        d = self.fc_ctx(d)                                                          # [Q x D x Td x H x 2] --> [Q x D x Td x H x 1]
        d = d.squeeze(dim=-1)                                                       # [Q x D x Td x H x 1] --> [Q x D x Td x H]
        d = d.unsqueeze(dim=2)                                                      # [Q x D x Td x H] --> [Q x D x 1 x Td x H]
        y = self.cosine_sim(q, d)                                                   # [Q x 1 x Tq x 1 x H], [Q x D x 1 x Td x H] --> [Q x D x Tq x Td]
        y = self.rbf_kernel(y, mask_d)                                              # [Q x D x Tq x Td] --> [Q x D x Tq]
        mask_q = mask_q.unsqueeze(1)                                                # [Q x Tq] --> [Q x 1 x Tq]
        y = y * mask_q                                                              # [Q x D x Tq], [Q x 1 x Tq] --> [Q x D x Tq]
        if not qti_mode:
            y = y.sum(dim=-1)                                                       # [Q x D x Tq] --> [Q x D]
        return y

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NDRM2(nn.Module):

    def __init__(self, args):
        super(NDRM2, self).__init__()
        self.args = args
        self.norm_dlen = BatchScale(1)
        self.norm_tf = BatchScale(1)
        self.fc_dlen = nn.Sequential(nn.Linear(1, 1), nn.ReLU())
        with torch.no_grad():
            self.fc_dlen[0].weight = nn.Parameter(torch.ones((1, 1), dtype=torch.float32), requires_grad=True)
            self.fc_dlen[0].bias = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

    def forward(self, qd, mask_q, q_idf, dlen, qti_mode=False):
        shape_dlen = dlen.shape
        dlen = dlen.view(-1, 1)                                                     # [Q x D] --> [QD x 1]
        dlen = self.norm_dlen(dlen)                                                 # [QD x 1] --> [QD x 1]
        dlen = dlen.view(shape_dlen + (1,))                                         # [QD x 1] --> [Q x D x 1]
        dlen = self.fc_dlen(dlen)                                                   # [Q x D x 1] --> [Q x D x 1]
        shape_qd = qd.shape
        y = qd.view(-1, 1)                                                          # [Q x D x Tq] --> [QDTqx 1]
        y = self.norm_tf(y)                                                         # [QDTq x 1] --> [QDTq x 1]
        y = y.view(shape_qd)                                                        # [QDTq x 1] --> [Q x D x Tq]
        y = y / (y + dlen + 1e-6)                                                   # [Q x D x Tq], [Q x D x 1] --> [Q x D x Tq]
        q_idf = q_idf.unsqueeze(dim=1)                                              # [Q x Tq] --> [Q x 1 x Tq]
        y = y * q_idf                                                               # [Q x D x Tq], [Q x 1 x Tq] --> [Q x D x Tq]
        mask_q = mask_q.unsqueeze(1)                                                # [Q x Tq] --> [Q x 1 x Tq]
        y = y * mask_q                                                              # [Q x D x Tq], [Q x 1 x Tq] --> [Q x D x Tq]
        if not qti_mode:
            y = y.sum(dim=-1)                                                       # [Q x D x Tq] --> [Q x D]
        return y

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NDRM3(nn.Module):

    def __init__(self, args):
        super(NDRM3, self).__init__()
        self.ndrm1 = NDRM1(args)
        self.ndrm2 = NDRM2(args)
        self.fc = nn.Sequential(nn.BatchNorm1d(2),
                                nn.Linear(2, 1))

    def forward(self, q, d, qd, mask_q, mask_d, q_idf, dlen, qti_mode=False):
        y_lat = self.ndrm1(q, d, mask_q, mask_d, qti_mode=True)                     # [*] --> [Q x D x Tq]
        y_exp = self.ndrm2(qd, mask_q, q_idf, dlen, qti_mode=True)                  # [*] --> [Q x D x Tq]
        y_lat = y_lat.unsqueeze(dim=-1)                                             # [Q x D x Tq] --> [Q x D x Tq x 1]
        y_exp = y_exp.unsqueeze(dim=-1)                                             # [Q x D x Tq] --> [Q x D x Tq x 1]
        y = torch.cat([y_lat, y_exp], dim=-1)                                       # [Q x D x Tq x 1], [Q x D x Tq x 1] --> [Q x D x Tq x 2]
        shape_y = y.shape
        y = y.view(-1, 2)                                                           # [Q x D x Tq x 2] --> [QDTq x 2]
        y = self.fc(y)                                                              # [QDTq x 2] --> [QDTq x 1]
        y = y.view(shape_y[:-1])                                                    # [QDTq x 1] --> [Q x D x Tq]
        mask_q = mask_q.unsqueeze(1)                                                # [Q x Tq] --> [Q x 1 x Tq]
        y = y * mask_q                                                              # [Q x D x Tq], [Q x 1 x Tq] --> [Q x D x Tq]
        if not qti_mode:
            y = y.sum(dim=-1)                                                       # [Q x D x Tq] --> [Q x D]
        return y

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RBFKernel(nn.Module):

    def __init__(self, args):
        super(RBFKernel, self).__init__()
        self.args = args
        mus = nn.Parameter(torch.from_numpy(np.array([(1 - 2 * i / self.args.rbf_kernel_dim) for i in range(self.args.rbf_kernel_dim)], dtype=np.float32)).view(1, 1, 1, 1, -1), requires_grad=False)
        sigmas = nn.Parameter(torch.from_numpy(np.array([0.1], dtype=np.float32)).view(1, 1, 1, 1, 1), requires_grad=False)
        denom = 2 * torch.pow(sigmas, 2)
        self.avg_pool = nn.AvgPool1d(self.args.rbf_kernel_pool_size, stride=self.args.rbf_kernel_pool_stride)
        self.fc = nn.Sequential(nn.Linear(self.args.rbf_kernel_dim, self.args.num_hidden_nodes),
                                nn.ReLU(),
                                nn.Dropout(p=self.args.dropout),
                                nn.Linear(self.args.num_hidden_nodes, 1))
        self.register_buffer('mus', mus)
        self.register_buffer('denom', denom)

    def forward(self, x, mask_d):
        y = x.unsqueeze(dim=-1)                                                     # [Q x D x Tq x Td] --> [Q x D x Tq x Td x 1]
        y = y - self.mus                                                            # [Q x D x Tq x Td x 1], [1 x 1 x 1 x 1 x K] --> [Q x D x Tq x Td x K]
        y = torch.pow(y, 2)                                                         # [Q x D x Tq x Td x K] --> [Q x D x Tq x Td x K]
        y = y / self.denom                                                          # [Q x D x Tq x Td x K], [1 x 1 x 1 x 1 x 1] --> [Q x D x Tq x Td x K]
        y = torch.exp(-y)                                                           # [Q x D x Tq x Td x K] --> [Q x D x Tq x Td x K]
        mask_d = mask_d.unsqueeze(dim=2)                                            # [Q x D x Td] --> [Q x D x 1 x Td]
        mask_d = mask_d.unsqueeze(dim=-1)                                           # [Q x D x 1 x Td] --> [Q x D x 1 x Td x 1]
        y = y * mask_d                                                              # [Q x D x Tq x Td x K], [Q x D x 1 x Td x 1] --> [Q x D x Tq x Td x K]
        shape_y = y.shape
        y = y.view(-1, shape_y[-2], shape_y[-1])                                    # [Q x D x Tq x Td x K] --> [QDTq x Td x K]
        y = y.permute(0, 2, 1)                                                      # [QDTq x Td x K] --> [QDTq x K x Td]
        y = self.avg_pool(y)                                                        # [QDTq x K x Td] --> [QDTq x K x N]
        y = y * self.args.rbf_kernel_pool_size                                      # [QDTq x K x N] --> [QDTq x K x N]
        y = torch.log(y + 1e-6)                                                     # [QDTq x K x N] --> [QDTq x K x N]
        y = y.permute(0, 2, 1)                                                      # [QDTq x K x N] --> [QDTq x N x K]
        y = self.fc(y)                                                              # [QDTq x N x K] --> [QDTq x N x 1]
        y, _ = y.max(dim=1)                                                         # [QDTq x N x 1] --> [QDTq x 1]
        y = y.view(shape_y[:3])                                                     # [QDTq x 1] --> [Q x D x Tq]
        return y


class BatchScale(nn.Module):

    def __init__(self, num_features):
        super(BatchScale, self).__init__()
        self.num_features = num_features
        self.register_buffer('running_mean', torch.zeros(num_features, dtype=torch.float32))
        self.register_buffer('num_samples', torch.tensor(0, dtype=torch.float32))

    def forward(self, x):
        if self.training:
            mb_size = x.shape[0]
            self.num_samples.add_(mb_size)                                          # [1] --> [1]
            new_mean = x.detach().sum(dim=0)                                        # [B x N] --> [N]
            delta = new_mean - (mb_size * self.running_mean)                        # [N], [N] --> [N]
            delta = delta / self.num_samples                                        # [N], [N] --> [N]
            self.running_mean.add_(delta)                                           # [N], [N] --> [N]
        y = x / (self.running_mean + 1e-6)                                          # [B x N], [N] --> [B x N]
        return y
