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
import os
import scipy.io as sio
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


args = []

class Spec(object):
    pass

def get_data_spec(model_name):
    if model_name in ['mnist', 'madry_mnist']:
        spec = Spec()
        spec.crop_size = 28
        spec.channels = 1
        spec.mean = 0.
        spec.rescale = [0., 1.]
        return spec
    else:
        raise

def get_model_path(model_path, model_name):
    return model_path + '/' + model_name

class MnistTestProducer(object):
    def __init__(self, data_type, index, target_label):
        import torchvision.transforms as transforms
        import torchvision
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0,), (1,))])

        testset = torchvision.datasets.MNIST(root='../data', train=False,
                                               download=True, transform=transform)
        

        test_data = testset.test_data.numpy() / 255.0
        test_labels = testset.test_labels.numpy()
        
        if data_type == 'raw':
            test_data = testset.test_data.numpy() / 255.0
            image = test_data[index:index+1]
            image = np.expand_dims( image, 0)
            image = image.astype(np.float32)
        elif data_type == 'fgsm':
            image = np.load('boundry/data/fgsm_model3_1023.npy')
            image = image.transpose(0, 3, 1, 2)
        elif data_type == 'cw':
            image = np.load('boundry/data/CW_model3_1023.npy')
            # image = image.transpose(0, 3, 1, 2)
        elif data_type == 'stn':
            data = sio.loadmat('/raid/chaowei/stn/mnist/result_testmodel3.mat')
            image = data['advimg'][target_label][index:index+1]
            image = image.transpose(0, 3, 1, 2)
        self.image = image[0].transpose(1,2,0)
        
        self.label = test_labels[index]
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
        perturb_images_var = Variable(torch.from_numpy(perturb_images).cuda(), requires_grad = False)
        prob = model(perturb_images_var)
        # st()
        _, pred = torch.max( prob, 1)
        pred = pred.cpu().data.numpy()
        for i in range(batch_size):
            result += [((r + step * i) * rmse - bias, pred[i])]

        r += batch_size * step
        citr += 1
        ctime = time.time()

def dotest(output_name,
           image_producer,
           model,
           batch_size):
    result = []
    image, label, index, data_type = image_producer.get_image()

    base = 'boundry_tmp/gradients/' 
    u = pickle.load(open(base + '{}_{}_{}_{}.pickle'.format(data_type, model.name, index, label)))
    # normolization 
    u /= np.sqrt(np.sum(u ** 2))
    # if 0:
    if os.path.isfile(base+'{}_{}_{}_{}.v.pickle'.format(data_type, model.name, index, label)):
        v = pickle.load(open( base+'{}_{}_{}_{}.v.pickle'.format(data_type, model.name, index, label)  ))
    else:
        while True:
            uu = u.reshape(28 * 28 * 1)
            v = np.random.normal(0, 1, 28 * 28 * 1)
            v = (v - np.dot(v, uu) * uu)
            v /= np.sqrt(np.sum(v * v))
            print 'uu dot v', np.sum(uu * v) # , v
            if np.abs(np.sum(uu * v)) < 1e-5:
                v = v.reshape((28, 28, 1))
                break
        pickle.dump(v, open(base+'{}_{}_{}_{}.v.pickle'.format(data_type, model.name, index, label), 'w'))

    samples = 255 / 0.2

    rmse = np.sqrt((u ** 2).mean())
    for i in tqdm.trange(-int(samples), int(samples)):
        assert u.shape == v.shape
        x = u * i * 0.2 / rmse
        a = search_direction(model, x, v, image, label, batch_size)
        # a = search_direction(sess, model, x, v, image, label, batch_size)
        result += [(i * 0.2, t[0], t[1]) for t in a]

    if output_name:
        outfile = '/raid/chaowei/stn/boundry_tmp/swipe-for-{}-{}-{}-{}.pickle'.format(data_type, model.name, index, label)
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
    torch.manual_seed(100)
    target_model = model_name
    if target_model == "model1":
        net = Net().cuda()
    elif target_model == "model2":
        net = Net2().cuda()
    elif target_model == "model3":
        net = Net3().cuda()
    else:
        print "target model error, {}".format(opt.target_model)
    net.eval()
    # net = Net3().cuda()
    # net.eval()
    checkpoint = torch.load('save/checkpoint_{}.pth.tar'.format(net.name))
    net.load_state_dict(checkpoint['state_dict'])

    dotest(output_file_dir,
           image_producer,
           net,
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
    parser.add_argument('--model', type=str, default='model3',
                        help='Models to be evaluated.')
    # parser.add_argument('--model_path', type=str, default='./models/',
                        # help='The base path to search for model files')
    parser.add_argument('--index', type=int, default='2000',
                        help='Evaluate a specific image by index.')
    parser.add_argument('--batch_size', type=int, default=2550,
                        help='Learning rate of each iteration.')
    parser.add_argument('--data_type', type=str, default='stn')
    parser.add_argument('--target_label', type=int, default=2)
    global args
    args = parser.parse_args()
    print 'args', args
    valid_idxs = np.load('boundry_tmp/valid_idx.npy')
    for i in valid_idxs:
        args.index = i
        for model in ['model1', 'model2', 'model3']:
            args.model = model
            for data_type in ['raw', 'fgsm', 'cw', 'stn']:
            # for data_type in ['fgsm']:
                args.data_type = data_type
            # data_spec = get_data_spec(model_name=args.model)
                image_producer = MnistTestProducer( data_type = args.data_type, index=args.index, target_label = args.target_label)

                exp1(args, image_producer)

if __name__ == '__main__':
    main()