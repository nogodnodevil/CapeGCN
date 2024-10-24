import argparse
import pickle
import sys
import time
from utils import Data, split_validation
from model import *
import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Cell_Phones_and_Accessories',
                    help='dataset name: Cell_Phones_and_Accessories/Grocery_and_Gourmet_Food/Sports_and_Outdoors/sample')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size 512')
parser.add_argument('--emb_size', type=int, default=64, help='embedding size')
parser.add_argument('--img_emb_size', type=int, default=64, help='image embedding size 64')
parser.add_argument('--text_emb_size', type=int, default=64, help='text embedding size 64')
parser.add_argument('--layers', type=int, default=1, help='the number of stacked layers')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--soft_lambda', type=float, default=1, help='weight of soft prompt')
parser.add_argument('--theta', type=float, default=0.0001, help='weight of contrastive learning loss')
parser.add_argument('--gpu', type=int, default=1, help='the ID of the graphics card')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args()
print(opt)
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists(ROOT_PATH + '/log'):
    os.mkdir(ROOT_PATH + '/log')
log_path = (ROOT_PATH + '/log/' + opt.dataset + '_' + str(opt.lr) + '_' + str(opt.layers) + '_' + str(opt.soft_lambda)
            + '_' + str(opt.theta) + '_' + str(opt.dropout) + '.txt')
f = open(log_path,'w')
print(opt, file=f, flush=True)

if __name__ == '__main__':
    train_data = pickle.load(open(ROOT_PATH + '/datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open(ROOT_PATH + '/datasets/' + opt.dataset + '/test.txt', 'rb'))

    if opt.dataset == 'Grocery_and_Gourmet_Food':
        n_node = 11638
    elif opt.dataset == 'Cell_Phones_and_Accessories':
        n_node = 8614
    elif opt.dataset == 'Sports_and_Outdoors':
        n_node = 18796
    else:
        print("Unkonwn dataset!")
        sys.exit()

    train_data = Data(train_data, shuffle=True, n_node=n_node)
    test_data = Data(test_data, shuffle=False, n_node=n_node)

    model = trans_to_cuda(CapeGCN(opt, n_node, train_data.adjacency, ROOT_PATH=ROOT_PATH))

    top_K = [1, 5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]

    start = time.time()
    for epoch in range(opt.epoch):
        print('-' * 80)
        print('epoch: ', epoch)
        print('-' * 33, file=f, flush=True)
        print(f'epoch: {epoch}', file=f, flush=True)
        metrics, total_loss = train_test(model, train_data, test_data, f)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            metrics['ndcg%d' % K] = np.mean(metrics['ndcg%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
            if best_results['metric%d' % K][2] < metrics['ndcg%d' % K]:
                best_results['metric%d' % K][2] = metrics['ndcg%d' % K]
                best_results['epoch%d' % K][2] = epoch
        print(metrics)
        print("%-5s\t %-5s\t %-5s\t %-5s" % ('P@10', 'M@10', 'P@20', 'M@20'))
        print("%-5.2f\t %-5.2f\t %-5.2f\t %-5.2f" % (best_results['metric10'][0], best_results['metric10'][1],
                                                 best_results['metric20'][0], best_results['metric20'][1]))
        print("%-5d\t %-5d\t %-5d\t %-5d" % (best_results['epoch10'][0], best_results['epoch10'][1],
                                             best_results['epoch20'][0], best_results['epoch20'][1]))

        print("%-5s\t %-5s\t %-5s\t %-5s" % ('P@10', 'M@10', 'P@20', 'M@20'), file=f, flush=True)
        print("%-5.2f\t %-5.2f\t %-5.2f\t %-5.2f" % (best_results['metric10'][0], best_results['metric10'][1],
                                                     best_results['metric20'][0], best_results['metric20'][1]), file=f, flush=True)
        print("%-5d\t %-5d\t %-5d\t %-5d" % (best_results['epoch10'][0], best_results['epoch10'][1],
                                             best_results['epoch20'][0], best_results['epoch20'][1]), file=f, flush=True)


    print('-------------------------------------------------------')
    print('-------------------------------------------------------', file=f, flush=True)
    end = time.time()
    print("Run time: %f s" % (end - start))
    print("Run time: %f s" % (end - start), file=f, flush=True)
    print(opt, file=f, flush=True)
