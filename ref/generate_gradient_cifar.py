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
import scipy.io as sio

import torch.backends.cudnn as cudnn
import models.cifar as models
import os 

# usage python generate_gradient.py 1518
# index = int(sys.argv[1])
# target_model = sys.argv[2]
# data_type = sys.argv[3]

index = int(0)
target_model = 'wrn'
data_type = 'raw'
num_classes = 10
for data_type in ['raw', 'fgsm', 'cw', 'stn']:
    batch_size = 1
    target_label = 2
    data = sio.loadmat('test_cifar_exp/cifar2_wrn34.mat')
    test_data = data['source']
    if data_type == 'raw':
        image = test_data[index:index+1]
        # image = np.expand_dims( image, 0)
        image = image.transpose(0, 3, 1, 2)
        # label = test_labels[index:index+1]
        image = image.astype(np.float32)
    elif data_type == 'fgsm':
        image = np.load('boundry/data/fgsm_wrn_0_cifar.npy')
        image = image.transpose(0, 3, 1, 2)
    elif data_type == 'cw':
        image = np.load('boundry/data/CW_wrn_0_cifar.npy')
        # image = image.transpose(0, 3, 1, 2)
    elif data_type == 'stn':
        stn_adv = data['advimg']
        image = stn_adv[index * 10 + target_label:index * 10 + target_label+ 1]
        image = image.transpose(0, 3, 1, 2)
    label = np.array([index % 10])

    # testiter = iter(testloader)
    # data, label = testiter.next()
    # # opt = parse_opt()
    # # opt.cuda = not opt.no_cuda and torch.cuda.is_available()
    # # opt.batch_size = 1
    torch.manual_seed(100)
    if target_model == 'resnet':
        arch = target_model
        depth = 32
        num_classes = num_classes
        model = models.__dict__[arch](
            num_classes=num_classes,
            depth=depth,
        )
        resume = 'checkpoint/resnet-32/checkpoint.pth.tar'
    elif target_model == 'wrn':
        depth = 34
        widen_factor = 10
        drop = 0.3
        arch = target_model
        num_classes = num_classes
        model = models.__dict__[arch](
                    num_classes=num_classes,
                    depth=depth,
                    widen_factor=widen_factor,
                    dropRate=drop,
                )
        resume = 'checkpoint/WRN-34-10/checkpoint.pth.tar'
    else:
        print "target model error, {}".format(target_model)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    # Resume
    title = 'cifar-10-' + target_model
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

    image_var = Variable(torch.from_numpy(image).cuda(), requires_grad=True )
    label_var = Variable(torch.from_numpy(label).cuda(),  requires_grad=False)
    criterion = nn.CrossEntropyLoss()
    output = model(image_var)
    loss = criterion(output, label_var)
    loss.backward()
    g = image_var.grad
    g = g.data.cpu().numpy()
    g = g.transpose(0, 2,3,1)
    g = g.squeeze(axis=0)
    filename = 'boundry/gradients/{}_{}_{}_{}_cifar.pickle'.format(data_type,  target_model, index, label[0])
    print filename
    pickle.dump(g, open(filename, 'w'), pickle.HIGHEST_PROTOCOL)
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
# images = mnist.test.images[index, None]
# labels = mnist.test.labels[index, None]

# X = tf.placeholder(tf.float32, [None, 28 * 28])
# import madry_mnist.model
# m = madry_mnist.model.Model(X)
# grad, = tf.gradients(m.xent, X)

# sess = tf.Session()

# checkpoint = tf.train.latest_checkpoint('models/madry_mnist')
# saver = tf.train.Saver()
# saver.restore(sess, checkpoint)

# g = sess.run(grad, feed_dict={
#     X: images,
#     m.y_input: labels
# })
# g = g.squeeze(axis=0).reshape(28, 28, 1)