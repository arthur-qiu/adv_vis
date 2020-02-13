# -*- coding: utf-8 -*-
import numpy as np
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
import json
from attack_methods import pgd
from models.wrn import WideResNet
from option import BaseOptions
import pickle
import tqdm
import time
import matplotlib.pyplot as plt

def search_direction(model, x, v, image, label,
                     lr_search):
    ctime = time.time()
    rmse = np.sqrt((v ** 2).mean())
    r = 0.2 / rmse
    step = r
    citr = 0
    upper_bound = 255 / rmse * 2
    bias = 255
    last = True
    result = []
    while True:
        if r > upper_bound:
            return result
        arr = np.arange(0, lr_search, 1) * step + r - 255 / rmse
        modifier = np.array([(x + v * d) / 255. for d in arr])
        inputs = np.array([image for _ in arr])
        perturb_images = inputs + modifier
        perturb_images = np.clip(perturb_images, 0, 1)
        perturb_images = perturb_images.astype(dtype=np.float32)
        perturb_images = np.transpose(perturb_images, (0, 3, 1, 2))
        preds = []
        b_size = perturb_images.shape[0] / 25
        perturb_images_var = Variable(torch.from_numpy(perturb_images).cuda(), requires_grad=False)
        for i in range(25):
            tmp_perturb_images_var = perturb_images_var[i * b_size: (i + 1) * b_size]
            prob = model(tmp_perturb_images_var)
            _, pred = torch.max(prob, 1)
            pred = pred.cpu().data.numpy()
            preds.append(pred)
        # prob = model(perturb_images_var)
        preds = np.concatenate(preds, 0)
        # _, pred = torch.max( prob, 1)
        # pred = pred.cpu().data.numpy()
        for i in range(lr_search):
            result += [((r + step * i) * rmse - bias, preds[i])]

        r += lr_search * step
        citr += 1
        ctime = time.time()

class PrivateOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # WRN Architecture
        self.parser.add_argument('--layers', default=28, type=int, help='total number of layers')
        self.parser.add_argument('--widen-factor', default=10, type=int, help='widen factor')
        self.parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability')

        self.parser.add_argument('--lr_search', type=int, default=2550,
                            help='Learning rate of each iteration.')
        self.parser.set_defaults(batch_size=1)
        self.parser.set_defaults(test_bs=1)

opt = PrivateOptions().parse()

state = {k: v for k, v in opt._get_kwargs()}

torch.manual_seed(opt.random_seed)
np.random.seed(opt.random_seed)
cudnn.benchmark = True

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(
                                         mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(
                                         mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

if opt.dataset == 'cifar10':
    train_data = dset.CIFAR10(opt.dataroot, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10(opt.dataroot, train=False, transform=test_transform)
    num_classes = 10
else:
    train_data = dset.CIFAR100(opt.dataroot, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR100(opt.dataroot, train=False, transform=test_transform)
    num_classes = 100

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=opt.batch_size, shuffle=True,
    num_workers=opt.prefetch, pin_memory=torch.cuda.is_available())
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=opt.test_bs, shuffle=False,
    num_workers=opt.prefetch, pin_memory=torch.cuda.is_available())

# Create model
if opt.model == 'wrn':
    net = WideResNet(opt.layers, num_classes, opt.widen_factor, dropRate=opt.droprate)
else:
    assert False, opt.model + ' is not supported.'

if opt.ngpu > 0:
    net = torch.nn.DataParallel(net, device_ids=list(range(opt.ngpu)))
    net.cuda()
    torch.cuda.manual_seed(opt.random_seed)

# Restore model if desired
if opt.load != '' and os.path.isfile(opt.load):
    net.load_state_dict(torch.load(opt.load))
    print('Appointed Model Restored!')
else:
    raise Exception("Could not resume")

lr = state['learning_rate']
optimizer = torch.optim.SGD(
    net.parameters(), lr, momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)

adversary_test = pgd.PGD(epsilon=opt.epsilon * 2, num_steps=opt.test_num_steps, step_size=opt.test_step_size * 2).cuda()

net.eval()
data_type = 'pgd'
index = 0
for data, target in test_loader:
    index += 1
    data, target = data.cuda(), target.cuda()

    adv_data = adversary_test(net, data, target)

    # forward
    output = net(adv_data)
    loss = F.cross_entropy(output, target)
    loss.backward()

    g = adv_data.grad
    g = g.data.cpu().numpy()
    g = g.transpose(0, 2, 3, 1)
    g = g.squeeze(axis=0)
    label = target[0]
    filename = 'boundry/gradients/{}_{}_{}_{}_cifar.pickle'.format(data_type, opt.model, index, label)
    print(filename)
    pickle.dump(g, open(filename, 'w'), pickle.HIGHEST_PROTOCOL)


    result = []
    base = 'boundry/gradients/'
    u = pickle.load(open(base + '{}_{}_{}_{}_cifar.pickle'.format(data_type, opt.model, index, label)))
    # normolization
    u /= np.sqrt(np.sum(u ** 2))
    if os.path.isfile(base + '{}_{}_{}_{}_cifar.v.pickle'.format(data_type, opt.model, index, label)):
        v = pickle.load(open(base + '{}_{}_{}_{}_cifar.v.pickle'.format(data_type, opt.model, index, label)))
    else:
        while True:
            uu = u.reshape(32 * 32 * 3)
            v = np.random.normal(0, 1, 32 * 32 * 3)
            v = (v - np.dot(v, uu) * uu)
            v /= np.sqrt(np.sum(v * v))
            print('uu dot v', np.sum(uu * v))  # , v
            if np.abs(np.sum(uu * v)) < 1e-5:
                v = v.reshape((32, 32, 3))
                break
        pickle.dump(v, open(base + '{}_{}_{}_{}_cifar.v.pickle'.format(data_type, opt.model, index, label), 'w'))

    samples = 255 / 0.2

    rmse = np.sqrt((u ** 2).mean())
    temp_image = adv_data[0].transpose(1, 2, 0)
    for i in tqdm.trange(-int(samples), int(samples)):
        assert u.shape == v.shape
        x = u * i * 0.2 / rmse
        a = search_direction(net, x, v, temp_image, label, opt.lr_search)
        # a = search_direction(sess, model, x, v, image, label, batch_size)
        result += [(i * 0.2, t[0], t[1]) for t in a]

    outfile = 'boundry/swipe-for-{}-{}-{}-{}-cifar.pickle'.format(data_type, opt.model, index, label)
    pickle.dump(result, open(outfile, 'w'), pickle.HIGHEST_PROTOCOL)


    points = pickle.load(open(outfile))
    print('loading', opt.model, 'done.')

    data = {}
    data['mnist'] = points

    print(points[2550 * 1275 + 1274])

    m = range(10)

    bound = 255 * 10
    bias = 255 * 5
    grid = [[0 for i in range(bound)] for j in range(bound)]
    for d in points:
        x = int(d[0] * 5 + bias + 1e-3)
        y = int(d[1] * 5 + bias + 1e-3)
        if x >= 0 and x < bound and y >=0 and y < bound:
            grid[x][y] = m[d[2]]

    g = np.asarray([[x for x in row] for row in grid])
    middle = 255 * 5
    plt.figure()
    n, bins, patches = plt.hist(g.flatten(), 10, normed=1, facecolor='green', alpha=0.75)
    plt.savefig('boundry_figs/mnist/{}_model3_hist.png'.format(data_type), format='png')
    for bound in [40, 100, 500, 1000, middle]:
        plt.clf()
        plt.figure(figsize=(16,12))
        plt.matshow(g.T[middle - bound:middle + bound, middle - bound:middle + bound],
                    origin='lower', interpolation='bilinear',
                    vmax=len(m), vmin=0,
                    extent=[-bound / 5.0, bound / 5.0, -bound / 5.0, bound / 5.0]
                  )
        plt.xlim(-bound / 5.0, bound / 5.0)
        plt.ylim(-bound / 5.0, bound / 5.0)
        plt.plot([0, 0], [-255, 255], color='white')
        plt.plot([-255, 255], [0, 0], color='white')
        print("bound:{}, fname:{}".format(bound,data_type))
        plt.savefig('boundry_figs/mnist/{}_model3_{}.png'.format(data_type, bound), format='png')


    break



