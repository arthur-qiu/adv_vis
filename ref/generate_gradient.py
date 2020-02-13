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
# usage python generate_gradient.py 1518
# index = int(sys.argv[1])
# target_model = sys.argv[2]
# data_type = sys.argv[3]

index = int(2000)
target_model = 'model3'
data_type = 'raw'
for target_model in ['model1', 'model2', 'model3']:
    
    for data_type in ['raw', 'fgsm', 'cw', 'stn']:
        # valid_idxs = []
        for i in range(100):
    
            batch_size = 1
            target_label = 2
            index = i

            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0,), (1,))])
            testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
            if testset.test_labels.numpy()[index] == target_label:
                continue
            # valid_idxs.append(i)
            # continue
            if data_type == 'raw':
                test_data = testset.test_data.numpy() / 255.0
                image = test_data[index:index+1]
                image = np.expand_dims( image, 0)
                image = image.astype(np.float32)
            elif data_type == 'fgsm':
                image = np.load('boundry_tmp/data/fgsm_{}_{}.npy'.format(target_model, index))
                image = image.transpose(0, 3, 1, 2)
            elif data_type == 'cw':
                image = np.load('boundry_tmp/data/CW_{}_{}.npy'.format(target_model, index))
                # image = image.transpose(0, 3, 1, 2)
            elif data_type == 'stn':
                data = sio.loadmat('/raid/chaowei/stn/mnist/result_test{}.mat'.format(target_model))
                image = data['advimg'][target_label][index:index+1]
                image = image.transpose(0, 3, 1, 2)
            test_labels = testset.test_labels.numpy()
            label = test_labels[index:index+1]

            # testiter = iter(testloader)
            # data, label = testiter.next()
            # # opt = parse_opt()
            # # opt.cuda = not opt.no_cuda and torch.cuda.is_available()
            # # opt.batch_size = 1
            torch.manual_seed(100)
            if target_model == "model1":
                mnist_net = Net().cuda()
            elif target_model == "model2":
                mnist_net = Net2().cuda()
            elif target_model == "model3":
                mnist_net = Net3().cuda()
            else:
                print "target model error, {}".format(opt.target_model)
            mnist_net.eval()
            checkpoint = torch.load('save/checkpoint_{}.pth.tar'.format(mnist_net.name))
            mnist_net.load_state_dict(checkpoint['state_dict'])
            image_var = Variable(torch.from_numpy(image).cuda(), requires_grad=True )
            label_var = Variable(torch.from_numpy(label).cuda(),  requires_grad=False)
            criterion = nn.CrossEntropyLoss()
            output = mnist_net(image_var)
            loss = criterion(output, label_var)
            # st()
            loss.backward()
            g = image_var.grad
            g = g.data.cpu().numpy()
            g = g.transpose(0, 2,3,1)
            g = g.squeeze(axis=0)
            filename = 'boundry_tmp/gradients/{}_{}_{}_{}.pickle'.format(data_type,  mnist_net.name, index, label[0])
            print filename
            pickle.dump(g, open(filename, 'w'), pickle.HIGHEST_PROTOCOL)
        # np.save('boundry_tmp/valid_idx.npy', np.array(valid_idxs))
        # st()
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
