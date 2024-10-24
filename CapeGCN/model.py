import datetime
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

class MLPs(nn.Module):
    def __init__(self, input_size, out_size, dropout=0.2):
        super(MLPs, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.activate = torch.nn.Tanh()
        self.mlp_1 = nn.Linear(input_size, out_size)
        self.mlp_2 = nn.Linear(out_size, out_size)

    def forward(self, emb_trans):
        emb_trans = self.dropout(self.activate(self.mlp_1(emb_trans)))
        emb_trans = self.dropout(self.activate(self.mlp_2(emb_trans)))
        return emb_trans

class ItemConv(nn.Module):
    def __init__(self, layers, emb_size):
        super(ItemConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.w_item = nn.ModuleList([nn.Linear(self.emb_size, self.emb_size, bias=False) for i in range(self.layers)])

    def forward(self, adjacency, embedding):
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        for i in range(self.layers):
            item_embeddings = trans_to_cuda(self.w_item[i])(item_embeddings)
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            final.append(F.normalize(item_embeddings, dim=-1, p=2))
        item_embeddings = torch.stack(final, dim=0)
        item_embeddings = torch.mean(item_embeddings, dim=0)
        return item_embeddings

class PromptLearner(nn.Module):
    def __init__(self, emb_size, dropout=0.2):
        super(PromptLearner, self).__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.linear =  nn.Linear(self.emb_size, self.emb_size)

    def forward(self, feat1=None):
        return F.dropout(self.linear(feat1), self.dropout)

class CapeGCN(nn.Module):
    def __init__(self, opt, n_node, adjacency, ROOT_PATH):
        super(CapeGCN, self).__init__()
        self.n_node = n_node
        self.dataset = opt.dataset
        self.emb_size = opt.emb_size
        self.img_emb_size = opt.img_emb_size
        self.text_emb_size = opt.text_emb_size
        self.batch_size = opt.batch_size
        self.l2 = opt.l2
        self.lr = opt.lr
        self.dropout = opt.dropout
        self.layers = opt.layers
        self.soft_lambda = opt.soft_lambda
        self.theta = opt.theta

        self.w_k = 10
        self.adjacency = adjacency

        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.pos_embedding = nn.Embedding(2000, self.emb_size)

        self.ItemGraph = ItemConv(self.layers, self.emb_size)
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.w_i = nn.Linear(self.emb_size, self.emb_size)
        self.w_s = nn.Linear(self.emb_size, self.emb_size)
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)

        self.mlp_mix_trans = MLPs(self.emb_size * 3, self.emb_size, dropout=self.dropout)
        self.prompt_image = PromptLearner(self.emb_size, self.dropout)
        self.prompt_text = PromptLearner(self.emb_size, self.dropout)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)
        self.init_parameters()

        img_path = ROOT_PATH + '/datasets/' + self.dataset + '/imgMatrixpca.npy'
        imgWeights = np.load(img_path)
        self.image_embedding = nn.Embedding(self.n_node, self.img_emb_size)
        img_pre_weight = np.array(imgWeights)
        self.image_embedding.weight.data.copy_(torch.from_numpy(img_pre_weight))

        text_path = ROOT_PATH + '/datasets/' + self.dataset + '/textMatrixpca.npy'
        textWeights = np.load(text_path)
        self.text_embedding = nn.Embedding(self.n_node, self.text_emb_size)
        text_pre_weight = np.array(textWeights)
        self.text_embedding.weight.data.copy_(torch.from_numpy(text_pre_weight))

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def prompt_module(self, item_emb, image_emb, text_emb):
        prompt_image_emb = self.prompt_image(item_emb)
        prompt_text_emb = self.prompt_text(item_emb)

        prompt_image_emb = torch.matmul(prompt_image_emb, torch.matmul(prompt_image_emb.transpose(-1, -2), image_emb))
        prompt_text_emb = torch.matmul(prompt_text_emb, torch.matmul(prompt_text_emb.transpose(-1, -2), text_emb))

        re_image_emb = F.dropout(image_emb + self.soft_lambda * F.normalize(prompt_image_emb, p=2, dim=1), self.dropout)
        re_text_emb = F.dropout(text_emb + self.soft_lambda * F.normalize(prompt_text_emb, p=2, dim=1), self.dropout)

        return item_emb, re_image_emb, re_text_emb

    def generate_sess_emb(self, item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, seq_h], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        return select

    def contrastive(self, item_emb, image_emb, text_emb, re_item_emb, re_image_emb, re_text_emb):
        tau = 1
        num_neg = self.num_negatives = 100

        image_sim_mat = torch.matmul(image_emb, re_image_emb.permute(1, 0))
        image_sim_mat = image_sim_mat / tau
        image_sim_mat = F.softmax(image_sim_mat, dim=-1)
        image_sim_mat = torch.exp(image_sim_mat, out=None)
        topk_image_values, _ = torch.topk(image_sim_mat, k=num_neg, dim=1)
        image_loss_one = torch.sum(torch.log10(torch.diag(image_sim_mat)))
        image_loss_two = torch.sum(torch.log10(torch.sum(topk_image_values, 1)))
        image_con_loss = image_loss_two - image_loss_one

        text_sim_mat = torch.matmul(text_emb, re_text_emb.permute(1, 0))
        text_sim_mat = text_sim_mat / tau
        text_sim_mat = F.softmax(text_sim_mat, dim=-1)
        text_sim_mat = torch.exp(text_sim_mat, out=None)
        topk_text_values, _ = torch.topk(text_sim_mat, k=num_neg, dim=1)
        text_loss_one = torch.sum(torch.log10(torch.diag(text_sim_mat)))
        text_loss_two = torch.sum(torch.log10(torch.sum(topk_text_values, 1)))
        text_con_loss = text_loss_two - text_loss_one

        con_loss = image_con_loss + text_con_loss

        return con_loss

    def forward(self, session_item, session_len, reversed_sess_item, mask, tar):
        item_emb = self.embedding.weight
        image_emb = self.image_embedding.weight
        text_emb = self.text_embedding.weight

        re_item_emb, re_image_emb, re_text_emb = self.prompt_module(item_emb, image_emb, text_emb)
        con_loss = self.contrastive(item_emb, image_emb, text_emb, re_item_emb, re_image_emb, re_text_emb)

        item_emb = self.ItemGraph(self.adjacency, item_emb)
        image_emb = self.ItemGraph(self.adjacency, image_emb)
        text_emb = self.ItemGraph(self.adjacency, text_emb)

        mix_emb = torch.cat((item_emb, image_emb, text_emb), dim=-1)
        mix_emb = self.mlp_mix_trans(mix_emb)

        sess_emb_i = self.generate_sess_emb(mix_emb, session_item, session_len, reversed_sess_item, mask)
        sess_emb_i = self.w_k * F.normalize(sess_emb_i, dim=-1, p=2)
        item_emb = F.normalize(item_emb, dim=-1, p=2)
        scores_item = torch.mm(sess_emb_i, torch.transpose(item_emb, 1, 0))

        return scores_item, self.theta*con_loss

def forward(model, i, data):
    tar, session_len, session_item, reversed_sess_item, mask, diff_mask = data.get_slice(i)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    scores_item, con_loss = model(session_item, session_len, reversed_sess_item, mask, tar)
    return tar, scores_item, con_loss

def train_test(model, train_data, test_data, f):
    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    model.train()
    for i in tqdm(slices):
        model.zero_grad()
        targets, scores, con_loss = forward(model, i, train_data)
        loss = model.loss_function(scores, targets) + con_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    print('\tLoss:\t%.3f' % total_loss, file=f, flush=True)

    print('start predicting: ', datetime.datetime.now())
    top_K = [1, 5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
        metrics['ndcg%d' % K] = []
    slices = test_data.generate_batch(model.batch_size)
    model.eval()
    for i in tqdm(slices):
        tar, scores, con_loss = forward(model, i, test_data)
        scores = trans_to_cpu(scores).detach().numpy()
        tar = trans_to_cpu(tar).detach().numpy()
        index = np.argsort(-scores, 1)
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                    metrics['ndcg%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
                    metrics['ndcg%d' % K].append(1 / (np.log2(np.where(prediction == target)[0][0] + 2)))

    return metrics, total_loss
