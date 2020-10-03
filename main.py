import os
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from gensim.models import KeyedVectors

from config import Config
from model import DeepCoNNDataset, DeepCoNN


def date():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def predicting(model, dataloader):
    mse, sample_count = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            user_reviews, item_reviews, ratings = map(lambda x: x.to(config.device), batch)
            predict = model(user_reviews, item_reviews)
            mse += F.mse_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
    return mse / sample_count  # dataloader上的均方误差


def train(train_dataloader, valid_dataloader, model, config, model_path):
    train_mse = predicting(model, train_dataloader)
    valid_mse = predicting(model, valid_dataloader)
    print(f'{date()}## Start the training! Initial train mse {train_mse:.6f}, validation mse {valid_mse:.6f}')
    start_time = time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    torch.optim.lr_scheduler.ExponentialLR(opt, config.learning_rate_decay)

    best_loss, best_epoch = 100, 0
    for epoch in range(config.train_epochs):
        model.train()  # 将模型设置为训练状态
        total_loss, total_samples = 0, 0
        for batch in train_dataloader:
            user_reviews, item_reviews, ratings = map(lambda x: x.to(config.device), batch)
            predict = model(user_reviews, item_reviews)
            loss = F.mse_loss(predict, ratings, reduction='sum')  # 平方和误差
            opt.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播计算梯度
            opt.step()  # 根据梯度信息更新所有可训练参数

            total_loss += loss.item()
            total_samples += len(predict)

        model.eval()  # 停止训练状态
        valid_mse = predicting(model, valid_dataloader)
        train_loss = total_loss / total_samples
        print(f"{date()}#### Epoch {epoch:3d}; train mse {train_loss:.6f}; validation mse {valid_mse:.6f}")

        if best_loss > valid_mse:
            best_loss = valid_mse
            torch.save(model, model_path)

    end_time = time.perf_counter()
    print(f'{date()}## End of training! Time used {end_time - start_time:.0f} seconds.')


def test(dataloader, model_path):
    print(f'{date()}## Start the testing!')
    start_time = time.perf_counter()

    best_model = torch.load(model_path)
    test_loss = predicting(best_model, dataloader)

    end_time = time.perf_counter()
    print(f"{date()}## Test end, test ems is {test_loss:.6f}, time used {end_time - start_time:.0f} seconds.")


if __name__ == '__main__':
    config = Config()
    print(f'{date()}## Load embedding and data...')
    word2vec = KeyedVectors.load_word2vec_format('data/embedding/GoogleNews-vectors-negative300.bin', binary=True)

    train_dataset = DeepCoNNDataset('data/music/train.csv', word2vec, config)
    valid_dataset = DeepCoNNDataset('data/music/valid.csv', word2vec, config)
    test_dataset = DeepCoNNDataset('data/music/test.csv', word2vec, config)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    del train_dataset, valid_dataset, test_dataset

    model = DeepCoNN(config, word2vec).to(config.device)
    # model = torch.load('model/best_model.pt')  # 继续训练以前的模型
    del word2vec  # 节省空间
    os.makedirs('model', exist_ok=True)  # 文件夹不存在则创建
    train(train_dataloader, valid_dataloader, model, config, 'model/best_model.pt')
    test(test_dataloader, 'model/best_model.pt')
