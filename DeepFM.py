import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
from Data import DataLoaderKuaiRec
from Data import DataLoaderMovieLens
from torch.utils.data import DataLoader
import torch
from torch import nn


class DeepFM(nn.Module):

    def __init__(self, n_features, user_df, item_df, dim=128):
        super(DeepFM, self).__init__()
        # Randomly initialize the feature vectors of all features
        self.features = nn.Embedding(n_features, dim, max_norm=1)
        # Record index of user and item characteristics
        self.user_df = user_df
        self.item_df = item_df
        # Get the sum of the number of user and item feature categories
        total_neigbours = user_df.shape[1] + item_df.shape[1]
        # Initializing the MLP layer
        self.mlp_layer = self.__mlp(dim * total_neigbours)

    def __mlp(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid())

    # FM
    def FMcross(self, feature_embs):
        # feature_embs:[ batch_size, n_features, dim ]
        # [ batch_size, dim ]
        square_of_sum = torch.sum(feature_embs, dim=1) ** 2
        # [ batch_size, dim ]
        sum_of_square = torch.sum(feature_embs ** 2, dim=1)
        # [ batch_size, dim ]
        output = square_of_sum - sum_of_square
        # [ batch_size, 1 ]
        output = torch.sum(output, dim=1, keepdim=True)
        # [ batch_size, 1 ]
        output = 0.5 * output
        # [ batch_size ]
        return torch.squeeze(output)


    # DNN
    def Deep(self, feature_embs):
        # feature_embs:[ batch_size, n_features, dim ]
        # [ batch_size, total_neigbours * dim ]
        feature_embs = feature_embs.reshape((feature_embs.shape[0], -1))
        # [ batch_size, 1 ]
        output = self.mlp_layer(feature_embs)
        # [ batch_size ]
        return torch.squeeze(output)


    # Combine user and item characteristics
    def __getAllFeatures(self, u, i):
        users = torch.LongTensor(self.user_df.loc[u].values)
        items = torch.LongTensor(self.item_df.loc[i].values)
        all = torch.cat([users, items], dim=1)
        return all


    def forward(self, u, i):
        # Get the feature index of the user and the item when combined
        all_feature_index = self.__getAllFeatures(u, i)
        # Take out the feature vector
        all_feature_embs = self.features(all_feature_index)
        # [batch_size]
        fm_out = self.FMcross(all_feature_embs)
        # [batch_size]
        deep_out = self.Deep(all_feature_embs)
        # [batch_size]
        out = torch.sigmoid(fm_out + deep_out)
        return out


# Evaluation
def doEva(net, test_triple):
    d = torch.LongTensor(test_triple)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    with torch.no_grad():
        out = net(u, i)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    precision = precision_score(r, y_pred)
    recall = recall_score(r, y_pred)
    acc = accuracy_score(r, y_pred)
    return precision, recall, acc


def train(epochs=50, batchSize=1024, lr=0.001, dim=128, eva_per_epochs=1, need_eva=True):
    # Read data
    train_triples, test_triples, user_df, item_df, n_features = DataLoaderKuaiRec.read_data()
    # Initialization Model
    net = DeepFM(n_features, user_df, item_df, dim)
    # Define the loss function
    criterion = torch.nn.BCELoss()
    # Initialization Optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.3)
    # Start training
    for e in range(epochs):
        all_lose = 0
        for u, i, r in DataLoader(train_triples, batch_size=batchSize, shuffle=True):
            r = torch.FloatTensor(r.detach().numpy())
            optimizer.zero_grad()
            logits = net(u, i)
            loss = criterion(logits, r)
            all_lose += loss
            loss.backward()
            optimizer.step()
        print('epoch {},avg_loss={:.4f}'.format(e, all_lose / (len(train_triples) // batchSize)))
        # Evaluation Model
        if e % eva_per_epochs == 0 and need_eva:
            p, r, acc = doEva(net, train_triples)
            print('train:p:{:.4f}, r:{:.4f}, acc:{:.4f}'.format(p, r, acc))
            p, r, acc = doEva(net, test_triples)
            print('test:p:{:.4f}, r:{:.4f}, acc:{:.4f}'.format(p, r, acc))

    return net


if __name__ == '__main__':
    train()
