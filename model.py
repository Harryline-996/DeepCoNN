import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


class DeepCoNNDataset(Dataset):
    def __init__(self, data_path, word2vec, config):
        self.word2vec = word2vec
        self.config = config
        self.PAD_WORD_idx = self.word2vec.vocab[self.config.PAD_WORD].index
        df = pd.read_csv(data_path, header=None, names=['userID', 'itemID', 'review', 'rating'])
        df['review'] = df['review'].apply(self._review2id)  # 分词->数字
        self.null_idx = set()  # 暂存空样本的下标，最后删除他们
        user_reviews = self._get_user_reviews(df)  # 收集每个user的评论列表
        item_reviews = self._get_item_reviews(df)
        rating = torch.Tensor(df['rating'].to_list()).view(-1, 1)
        self.user_reviews = user_reviews[[idx for idx in range(user_reviews.shape[0]) if idx not in self.null_idx]]
        self.item_reviews = item_reviews[[idx for idx in range(item_reviews.shape[0]) if idx not in self.null_idx]]
        self.rating = rating[[idx for idx in range(rating.shape[0]) if idx not in self.null_idx]]

    def __getitem__(self, idx):
        return self.user_reviews[idx], self.item_reviews[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]

    def _get_user_reviews(self, df):
        # 对于每条训练数据，生成用户的所有评论汇总
        reviews_by_user = dict(list(df[['itemID', 'review']].groupby(df['userID'])))  # 每个用户的评论汇总
        user_reviews = []
        for idx, (uid, iid) in enumerate(zip(df['userID'], df['itemID'])):
            reviews = reviews_by_user[uid]  # 取出uid的所有评论：DataFrame
            reviews = reviews['review'][reviews['itemID'] != iid].to_list()  # 获取uid除了对当前item外的所有评论：列表
            if len(reviews) == 0:
                self.null_idx.add(idx)  # mark the index of null sample
            reviews = self._adjust_review_list(reviews, self.config.review_length, self.config.review_count)
            user_reviews.append(reviews)
        return torch.LongTensor(user_reviews)

    def _get_item_reviews(self, df):
        # 与get_user_reviews()同理
        reviews_by_item = dict(list(df[['userID', 'review']].groupby(df['itemID'])))  # 每个item的评论汇总
        item_reviews = []
        for idx, (uid, iid) in enumerate(zip(df['userID'], df['itemID'])):
            reviews = reviews_by_item[iid]  # 取出item的所有评论：DataFrame
            reviews = reviews['review'][reviews['userID'] != uid].to_list()  # 获取item除当前user外的所有评论：列表
            if len(reviews) == 0:
                self.null_idx.add(idx)
            reviews = self._adjust_review_list(reviews, self.config.review_length, self.config.review_count)
            item_reviews.append(reviews)
        return torch.LongTensor(item_reviews)

    def _adjust_review_list(self, reviews, r_length, r_count):
        reviews = reviews[:r_count] + [[self.PAD_WORD_idx] * r_length] * (r_count - len(reviews))  # 评论数量固定
        reviews = [r[:r_length] + [0] * (r_length - len(r)) for r in reviews]  # 每条评论定长
        return reviews

    def _review2id(self, review):
        #  将一个评论字符串分词并转为数字
        if not isinstance(review, str):
            return []  # 貌似pandas的一个bug，读取出来的评论如果是空字符串，review类型会变成float
        wids = []
        for word in review.split():
            if word in self.word2vec:
                wids.append(self.word2vec.vocab[word].index)  # 单词映射为数字
            else:
                wids.append(self.PAD_WORD_idx)
        return wids


class FactorizationMachine(nn.Module):

    def __init__(self, p, k):
        super(FactorizationMachine, self).__init__()

        self.p, self.k = p, k
        self.linear = nn.Linear(self.p, 1, bias=True)
        self.v = nn.Parameter(torch.zeros(self.p, self.k))
        self.drop = nn.Dropout(0.2)

    def fm_layer(self, x):
        linear_part = self.linear(x)
        inter_part1 = torch.mm(x, self.v)
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))
        pair_interactions = torch.sum(torch.sub(torch.pow(inter_part1, 2),
                                                inter_part2), dim=1)
        self.drop(pair_interactions)
        output = linear_part.transpose(1, 0) + 0.5 * pair_interactions
        return output

    def forward(self, x):
        output = self.fm_layer(x)
        return output.view(-1, 1)


class DeepCoNN(nn.Module):

    def __init__(self, config, word2vec):
        super(DeepCoNN, self).__init__()

        word_dim = word2vec.vector_size
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word2vec.vectors))

        self.conv_u = nn.Sequential(
            nn.Conv1d(
                in_channels=word_dim,
                out_channels=config.kernel_count,
                kernel_size=config.kernel_size,
                padding=(config.kernel_size - 1) // 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, config.review_length)),
            nn.Dropout(p=config.dropout_prob))
        self.linear_u = nn.Sequential(
            nn.Linear(config.kernel_count * config.review_count, config.cnn_out_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_prob))

        self.conv_i = nn.Sequential(
            nn.Conv1d(
                in_channels=word_dim,
                out_channels=config.kernel_count,
                kernel_size=config.kernel_size,
                padding=(config.kernel_size - 1) // 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, config.review_length)),
            nn.Dropout(p=config.dropout_prob))
        self.linear_i = nn.Sequential(
            nn.Linear(config.kernel_count * config.review_count, config.cnn_out_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_prob))

        self.out = FactorizationMachine(config.cnn_out_dim * 2, 10)

    def forward(self, user_review, item_review):  # 实际shape(batch_size, review_count, review_length)
        batch_size = user_review.shape[0]
        new_batch_size = user_review.shape[0] * user_review.shape[1]

        user_review = user_review.reshape(new_batch_size, -1)
        item_review = item_review.reshape(new_batch_size, -1)
        u_vec = self.embedding(user_review).permute(0, 2, 1)
        i_vec = self.embedding(item_review).permute(0, 2, 1)

        user_latent = self.conv_u(u_vec).reshape(batch_size, -1)
        item_latent = self.conv_i(i_vec).reshape(batch_size, -1)

        user_latent = self.linear_u(user_latent)
        item_latent = self.linear_i(item_latent)

        concat_latent = torch.cat((user_latent, item_latent), dim=1)
        prediction = self.out(concat_latent)
        return prediction
