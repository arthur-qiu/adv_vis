import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class SparsifyBase(nn.Module):
    def __init__(self, sparse_ratio=0.5):
        super(SparsifyBase, self).__init__()
        self.sr = sparse_ratio
        self.preact = None
        self.act = None

    def get_activation(self):
        def hook(model, input, output):
            self.preact = input[0].cpu().detach().clone()
            self.act = output.cpu().detach().clone()

        return hook

    def record_activation(self):
        self.register_forward_hook(self.get_activation())


class Sparsify1D(SparsifyBase):
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify1D, self).__init__()
        self.sr = sparse_ratio

    def forward(self, x):
        k = int(self.sr * x.shape[1])
        topval = x.topk(k, dim=1)[0][:, -1]
        topval = topval.expand(x.shape[1], x.shape[0]).permute(1, 0)
        comp = (x >= topval).to(x)
        return comp * x


class Sparsify1D_kactive(SparsifyBase):
    def __init__(self, k=1):
        super(Sparsify1D_kactive, self).__init__()
        self.k = k

    def forward(self, x):
        k = self.k
        topval = x.topk(k, dim=1)[0][:, -1]
        topval = topval.expand(x.shape[1], x.shape[0]).permute(1, 0)
        comp = (x >= topval).to(x)
        return comp * x


class Sparsify2D(SparsifyBase):
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D, self).__init__()
        self.sr = sparse_ratio

        self.preact = None
        self.act = None

    def forward(self, x):
        layer_size = x.shape[2] * x.shape[3]
        k = int(self.sr * layer_size)
        tmpx = x.view(x.shape[0], x.shape[1], -1)
        topval = tmpx.topk(k, dim=2)[0][:, :, -1]
        topval = topval.expand(x.shape[2], x.shape[3], x.shape[0], x.shape[1]).permute(2, 3, 0, 1)
        comp = (x >= topval).to(x)
        return comp * x


class Sparsify2D_vol(SparsifyBase):
    '''cross channel sparsify'''

    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D_vol, self).__init__()
        self.sr = sparse_ratio

    def forward(self, x):
        size = x.shape[1] * x.shape[2] * x.shape[3]
        k = int(self.sr * size)

        tmpx = x.view(x.shape[0], -1)
        topval = tmpx.topk(k, dim=1)[0][:, -1]
        topval = topval.repeat(tmpx.shape[1], 1).permute(1, 0).view_as(x)
        comp = (x >= topval).to(x)
        return comp * x


class Sparsify2D_kactive(SparsifyBase):
    '''cross channel sparsify'''

    def __init__(self, k):
        super(Sparsify2D_vol, self).__init__()
        self.k = k

    def forward(self, x):
        k = self.k
        tmpx = x.view(x.shape[0], -1)
        topval = tmpx.topk(k, dim=1)[0][:, -1]
        topval = topval.repeat(tmpx.shape[1], 1).permute(1, 0).view_as(x)
        comp = (x >= topval).to(x)
        return comp * x


class Sparsify2D_abs(SparsifyBase):
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D_abs, self).__init__()
        self.sr = sparse_ratio

    def forward(self, x):
        layer_size = x.shape[2] * x.shape[3]
        k = int(self.sr * layer_size)
        absx = torch.abs(x)
        tmpx = absx.view(absx.shape[0], absx.shape[1], -1)
        topval = tmpx.topk(k, dim=2)[0][:, :, -1]
        topval = topval.expand(absx.shape[2], absx.shape[3], absx.shape[0], absx.shape[1]).permute(2, 3, 0, 1)
        comp = (absx >= topval).to(x)
        return comp * x


class Sparsify2D_invabs(SparsifyBase):
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D_invabs, self).__init__()
        self.sr = sparse_ratio

    def forward(self, x):
        layer_size = x.shape[2] * x.shape[3]
        k = int(self.sr * layer_size)
        absx = torch.abs(x)
        tmpx = absx.view(absx.shape[0], absx.shape[1], -1)
        topval = tmpx.topk(k, dim=2, largest=False)[0][:, :, -1]
        topval = topval.expand(absx.shape[2], absx.shape[3], absx.shape[0], absx.shape[1]).permute(2, 3, 0, 1)
        comp = (absx >= topval).to(x)
        return comp * x


class breakReLU(nn.Module):
    def __init__(self, sparse_ratio=5):
        super(breakReLU, self).__init__()
        self.h = sparse_ratio
        self.thre = nn.Threshold(0, -self.h)

    def forward(self, x):
        return self.thre(x)


class SmallCNN(nn.Module):
    def __init__(self, fc_in=3136, n_classes=10):
        super(SmallCNN, self).__init__()

        self.module_list = nn.ModuleList([nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                                          nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                                          nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                          nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                                          Flatten(),
                                          nn.Linear(fc_in, 100), nn.ReLU(),
                                          nn.Linear(100, n_classes)])

    def forward(self, x):
        for i in range(len(self.module_list)):
            x = self.module_list[i](x)
        return x

    def forward_to(self, x, layer_i):
        for i in range(layer_i):
            x = self.module_list[i](x)
        return x


sparse_func_dict = {
    'reg': Sparsify2D,  # top-k value
    'abs': Sparsify2D_abs,  # top-k absolute value
    'invabs': Sparsify2D_invabs,  # top-k minimal absolute value
    'vol': Sparsify2D_vol,  # cross channel top-k
    'brelu': breakReLU,  # break relu
    'kact': Sparsify2D_kactive,
    'relu': nn.ReLU
}


class SparseBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, sp=0.5, sp_func='reg', bias=True):
        super(SparseBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = sparse_func_dict[sp_func](sp)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = sparse_func_dict[sp_func](sp)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=bias)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=bias) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class SparseNetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, sp=0.5, sp_func='reg', bias=True):
        super(SparseNetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, sp, sp_func, bias)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, sp, sp_func, bias):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, sp, sp_func,
                      bias))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class SparseWideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, sp=0.5, sp_func='reg', bias=True):
        super(SparseWideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = SparseBasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=bias)
        # 1st block
        self.block1 = SparseNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, sp, sp_func, bias)
        # 2nd block
        self.block2 = SparseNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, sp, sp_func, bias)
        # 3rd block
        self.block3 = SparseNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, sp, sp_func, bias)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        # self.relu = nn.ReLU(inplace=True)
        self.sparse = sparse_func_dict[sp_func](sp)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if bias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.sparse(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

class SparseWideResNet_Feat(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, sp=0.5, sp_func='reg', bias=True):
        super(SparseWideResNet_Feat, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = SparseBasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=bias)
        # 1st block
        self.block1 = SparseNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, sp, sp_func, bias)
        # 2nd block
        self.block2 = SparseNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, sp, sp_func, bias)
        # 3rd block
        self.block3 = SparseNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, sp, sp_func, bias)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        # self.relu = nn.ReLU(inplace=True)
        self.sparse = sparse_func_dict[sp_func](sp)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if bias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.sparse(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return out, self.fc(out)