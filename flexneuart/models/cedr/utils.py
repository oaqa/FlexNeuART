#
# This code is borrowed from CEDR: https://github.com/Georgetown-IR-Lab/cedr
#
# (c) Georgetown IR lab
#
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#
import torch

class PACRRConvMax2dModule(torch.nn.Module):

    def __init__(self, shape, n_filters, k, channels):
        super().__init__()
        self.shape = shape
        if shape != 1:
            self.pad = torch.nn.ConstantPad2d((0, shape-1, 0, shape-1), 0)
        else:
            self.pad = None
        self.conv = torch.nn.Conv2d(channels, n_filters, shape)
        self.activation = torch.nn.ReLU()
        self.k = k
        self.shape = shape
        self.channels = channels

    def forward(self, simmat):
        BATCH, CHANNELS, QLEN, DLEN = simmat.shape
        if self.pad:
            simmat = self.pad(simmat)
        conv = self.activation(self.conv(simmat))
        top_filters, _ = conv.max(dim=1)
        # LB: This a work around for rarely occurring weird cases of very short documents
        if DLEN < self.k:
            # padding with zeros the last dim to make it have it at least DLEN elements
            top_filters = torch.nn.functional.pad(top_filters, (0, self.k - DLEN))
        top_toks, _ = top_filters.topk(self.k, dim=2)
        result = top_toks.reshape(BATCH, QLEN, self.k)
        return result


class SimmatModule(torch.nn.Module):

    def __init__(self, padding=-1):
        super().__init__()
        self.padding = padding
        self._hamming_index_loaded = None
        self._hamming_index = None

    def forward(self, query_embed, doc_embed, query_tok, doc_tok):
        simmat = []

        for a_emb, b_emb in zip(query_embed, doc_embed):
            BAT, A, B = a_emb.shape[0], a_emb.shape[1], b_emb.shape[1]
            # embeddings -- cosine similarity matrix
            a_denom = a_emb.norm(p=2, dim=2).reshape(BAT, A, 1).expand(BAT, A, B) + 1e-9 # avoid 0div
            b_denom = b_emb.norm(p=2, dim=2).reshape(BAT, 1, B).expand(BAT, A, B) + 1e-9 # avoid 0div
            perm = b_emb.permute(0, 2, 1)
            sim = a_emb.bmm(perm)
            sim = sim / (a_denom * b_denom)

            # nullify padding (indicated by -1 by default)
            nul = torch.zeros_like(sim)
            sim = torch.where(query_tok.reshape(BAT, A, 1).expand(BAT, A, B) == self.padding, nul, sim)
            sim = torch.where(doc_tok.reshape(BAT, 1, B).expand(BAT, A, B) == self.padding, nul, sim)

            simmat.append(sim)
        return torch.stack(simmat, dim=1)


class DRMMLogCountHistogram(torch.nn.Module):
    def __init__(self, bins):
        super().__init__()
        self.bins = bins

    def forward(self, simmat, dtoks, qtoks):
        # THIS IS SLOW ... Any way to make this faster? Maybe it's not worth doing on GPU?
        BATCH, CHANNELS, QLEN, DLEN = simmat.shape
        # +1e-5 to nudge scores of 1 to above threshold
        bins = ((simmat + 1.000001) / 2. * (self.bins - 1)).int()
        # set weights of 0 for padding (in both query and doc dims)
        weights = ((dtoks != -1).reshape(BATCH, 1, DLEN).expand(BATCH, QLEN, DLEN) * \
                  (qtoks != -1).reshape(BATCH, QLEN, 1).expand(BATCH, QLEN, DLEN)).float()

        # no way to batch this... loses gradients here. https://discuss.pytorch.org/t/histogram-function-in-pytorch/5350
        bins, weights = bins.cpu(), weights.cpu()
        histogram = []
        for superbins, w in zip(bins, weights):
            result = []
            for b in superbins:
                result.append(torch.stack([torch.bincount(q, x, self.bins) for q, x in zip(b, w)], dim=0))
            result = torch.stack(result, dim=0)
            histogram.append(result)
        histogram = torch.stack(histogram, dim=0)

        # back to GPU
        histogram = histogram.to(simmat.device)
        return (histogram.float() + 1e-5).log()


class KNRMRbfKernelBank(torch.nn.Module):
    def __init__(self, mus=None, sigmas=None, dim=1, requires_grad=True):
        super().__init__()
        self.dim = dim
        kernels = [KNRMRbfKernel(m, s, requires_grad=requires_grad) for m, s in zip(mus, sigmas)]
        self.kernels = torch.nn.ModuleList(kernels)

    def count(self):
        return len(self.kernels)

    def forward(self, data):
        return torch.stack([k(data) for k in self.kernels], dim=self.dim)


class KNRMRbfKernel(torch.nn.Module):
    def __init__(self, initial_mu, initial_sigma, requires_grad=True):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.tensor(initial_mu), requires_grad=requires_grad)
        self.sigma = torch.nn.Parameter(torch.tensor(initial_sigma), requires_grad=requires_grad)

    def forward(self, data):
        adj = data - self.mu
        return torch.exp(-0.5 * adj * adj / self.sigma / self.sigma)
