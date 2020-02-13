import argparse
import os, os.path, time
import sys
import numpy as np
# import tensorflow as tf
import cPickle as pickle
import tqdm
import sys
import cPickle as pickle
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pdb
from pdb import set_trace as st
from mnist_model import Net, Net2, Net3

import torch.backends.cudnn as cudnn
import models.cifar as models
import os 

import scipy.io as sio
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


args = []

class Spec(object):
    pass

def get_data_spec(model_name):
    spec = Spec()
    spec.crop_size = 32
    spec.channels = 3
    spec.mean = 0.
    spec.rescale = [0., 1.]
    return spec

def get_model_path(model_path, model_name):
    return model_path + '/' + model_name

class CIFARTestProducer(object):
    def __init__(self, data_type, index, target_label):
        
        data = sio.loadmat('test_cifar_exp/cifar2_resnet.mat')
        test_data = data['source']
        if data_type == 'raw':
            image = test_data[index:index+1]
            # image = np.expand_dims( image, 0)
            image = image.transpose(0, 3, 1, 2)
            # label = test_labels[index:index+1]
            image = image.astype(np.float32)
        elif data_type == 'fgsm':
            image = np.load('boundry_tmp/data/fgsm_resnet_6_cifar.npy')
            # image = image.transpose(0, 3, 1, 2)
        elif data_type == 'cw':
            image = np.load('boundry_tmp/data/CW_resnet_6_cifar.npy')
            # image = image.transpose(0, 3, 1, 2)
        elif data_type == 'stn':
            stn_adv = data['advimg']
            # image = stn_adv[index * 10 + target_label:index * 10 + target_label+ 1]
            image = stn_adv[index:index+1]
            image = image.transpose(0, 3, 1, 2)
            image = image.astype(np.float32)

            # image = image.transpose(0, 3, 1, 2)
        if data_type == 'raw':
            self.label = index // 10
        else:
            self.label = index % 10
        self.image = image[0].transpose(1,2,0)
        # self.label = index % 10
        self.data_type = data_type
        self.index = index

    def get_image(self):
        image = self.image
        label = self.label
        # name = str(self.index)
        index = self.index
        data_type = self.data_type
        return image, label, index, data_type 


def search_direction(model, x, v, image, label,
                     batch_size):
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
        arr = np.arange(0, batch_size, 1) * step + r - 255 / rmse
        modifier = np.array([(x + v * d) / 255. for d in arr])
        # feed_dict = {model["inputs"]: [image for _ in arr],
        #              model["modifier"]: modifier}
        inputs = np.array([ image for _ in arr])
        perturb_images = inputs + modifier
        perturb_images = np.clip(perturb_images, 0, 1)
        perturb_images = perturb_images.astype( dtype=np.float32)
        perturb_images = np.transpose(perturb_images, (0, 3, 1, 2))
        preds = []
        b_size = perturb_images.shape[0] / 10
        perturb_images_var = Variable(torch.from_numpy(perturb_images).cuda(), requires_grad = False)
        for i in range(10):
            tmp_perturb_images_var = perturb_images_var[i* b_size: (i+1) * b_size]
            prob = model(tmp_perturb_images_var)
            _, pred = torch.max( prob, 1)
            pred = pred.cpu().data.numpy()
            preds.append(pred)
        # prob = model(perturb_images_var)
        preds = np.concatenate(preds, 0)
        # _, pred = torch.max( prob, 1)
        # pred = pred.cpu().data.numpy()
        for i in range(batch_size):
            result += [((r + step * i) * rmse - bias, preds[i])]

        r += batch_size * step
        citr += 1
        ctime = time.time()

def dotest(output_name,
           image_producer,
           model,
           batch_size):
    result = []
    image, label, index, data_type = image_producer.get_image()

    ####test 

    img_var = Variable(torch.from_numpy( np.array( [image.transpose(2,0,1)] ) ).cuda(), requires_grad = False)
    pred = model(img_var)
    print pred
    base = 'boundry/gradients/' 
    u = pickle.load(open(base + '{}_{}_{}_{}_cifar.pickle'.format(data_type, model.name, index, label)))
    # normolization 
    u /= np.sqrt(np.sum(u ** 2))
    if os.path.isfile(base+'{}_{}_{}_{}_cifar.v.pickle'.format(data_type, model.name, index, label)):
        v = pickle.load(open( base+'{}_{}_{}_{}_cifar.v.pickle'.format(data_type, model.name, index, label)  ))
    else:
        while True:
            uu = u.reshape(32 * 32 * 3)
            v = np.random.normal(0, 1, 32 * 32 * 3)
            v = (v - np.dot(v, uu) * uu)
            v /= np.sqrt(np.sum(v * v))
            print 'uu dot v', np.sum(uu * v) # , v
            if np.abs(np.sum(uu * v)) < 1e-5:
                v = v.reshape((32, 32, 3))
                break
        pickle.dump(v, open(base+'{}_{}_{}_{}_cifar.v.pickle'.format(data_type, model.name, index, label), 'w'))

    samples = 255 / 0.2

    rmse = np.sqrt((u ** 2).mean())
    for i in tqdm.trange(-int(samples), int(samples)):
        assert u.shape == v.shape
        x = u * i * 0.2 / rmse
        a = search_direction(model, x, v, image, label, batch_size)
        # a = search_direction(sess, model, x, v, image, label, batch_size)
        result += [(i * 0.2, t[0], t[1]) for t in a]

    if output_name:
        outfile = 'boundry/swipe-for-{}-{}-{}-{}-cifar.pickle'.format(data_type, model.name, index, label)
        pickle.dump(result, open(outfile, 'w'), pickle.HIGHEST_PROTOCOL)


def test(model_name,
         image_producer,
         output_file_dir=None,
         data_spec=None,
         batch_size=1):
    """Compute the gradients for the given network and images."""

    # model_test = build_evaluator(model_name,
    #                             batch_size,
    #                             data_spec,
    #                             adv_device)
    num_classes = 10
    torch.manual_seed(100)
    if model_name == 'resnet':
        arch = model_name
        depth = 32
        num_classes = num_classes
        model = models.__dict__[arch](
            num_classes=num_classes,
            depth=depth,
        )
        resume = 'checkpoint/resnet-32/checkpoint.pth.tar'
    elif model_name == 'wrn':
        depth = 34
        widen_factor = 10
        drop = 0.3
        arch = model_name
        num_classes = num_classes
        model = models.__dict__[arch](
                    num_classes=num_classes,
                    depth=depth,
                    widen_factor=widen_factor,
                    dropRate=drop,
                )
        resume = 'checkpoint/WRN-34-10/checkpoint.pth.tar'
    else:
        print "target model error, {}".format(model_name)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    # Resume
    title = 'cifar-10-' + model_name
    print title
        # Load checkpoint.
    print('==> Resuming from checkpoint..{}'.format(resume))
    assert os.path.isfile(resume), 'Error: no checkpoint directory found!'
    checkpoint = os.path.dirname(resume)
    checkpoint = torch.load(resume)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.name = model_name
    dotest(output_file_dir,
           image_producer,
           model,
           batch_size)

def exp1(args, image_producer, data_spec = None):
    test(args.model,
         image_producer,
         args.output_dir,
         # data_spec,
         batch_size=args.batch_size)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Evaluate model on some dataset.')
    # parser.add_argument('-i', '--input_dir', type=str, required=True,
    #                     help='Directory of dataset.')
    parser.add_argument('-o', '--output_dir', type=str, default='yes',
                        help='Whether to save the output. If not, just waste time. Always saves to current dir.')
    parser.add_argument('--model', type=str, default='resnet',
                        help='Models to be evaluated.')
    parser.add_argument('--index', type=int, default=36,
                        help='Evaluate a specific image by index.')
    parser.add_argument('--batch_size', type=int, default=2550,
                        help='Learning rate of each iteration.')
    parser.add_argument('--data_type', type=str, default='stn')
    parser.add_argument('--target_label', type=int, default=6)
    global args
    args = parser.parse_args()
    print 'args', args
    data_type = 'raw'
    for data_type in ['raw', 'fgsm', 'cw', 'stn']:
    # for data_type in ['stn']:
        args.data_type = data_type
    # data_spec = get_data_spec(model_name=args.model)
        image_producer = CIFARTestProducer( data_type = args.data_type, index=args.index, target_label = args.target_label)

        exp1(args, image_producer)

if __name__ == '__main__':
    main()