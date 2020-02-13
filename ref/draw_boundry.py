import cPickle as pk
import time
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from pdb import set_trace as st
model = 'mnist'
ctime = time.time()


# file_path = 'boundry/swipe-for-{}-wrn-0-0-cifar.pickle'#cifar 
file_path = 'boundry_tmp/swipe-for-{}-model3-4-4.pickle'

fnames = ['raw', 'fgsm', 'cw', 'stn']
# fnames = ['fgsm']
for fname in fnames:

    points = pk.load(open(file_path.format(fname)))
    print 'loading', model, 'done, using ', time.time() - ctime, 'sec.'
    data = {}
    data['mnist'] = points
    print points[2550 * 1275 + 1274]
    """
    s = set()

    for model in data:
        for d in data[model]:
            s.add(d[2])

    print 'size', len(s)
    m = {}
    m[393] = 0

    for c in s:
        if c != 393:
            m[c] = len(m)
    """
    m = range(10)

    for model in ['mnist']:

        bound = 255 * 10
        bias = 255 * 5
        grid = [[0 for i in range(bound)] for j in range(bound)]
        for d in data[model]:
            x = int(d[0] * 5 + bias + 1e-3)
            y = int(d[1] * 5 + bias + 1e-3)
            if x >= 0 and x < bound and y >=0 and y < bound:
                grid[x][y] = m[d[2]]

        g = np.asarray([[x for x in row] for row in grid])
        middle = 255 * 5
        plt.figure()
        n, bins, patches = plt.hist(g.flatten(), 10, normed=1, facecolor='green', alpha=0.75)
        plt.savefig('boundry_figs/mnist/{}_model3_hist.png'.format(fname), format='png')    
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
            print "bound:{}, fname:{}".format(bound,fname)
            plt.savefig('boundry_figs/mnist/{}_model3_{}.png'.format(fname, bound), format='png')
            # plt.show()

