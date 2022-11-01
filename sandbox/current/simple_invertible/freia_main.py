import torch.nn as nn
import torch
import FrEIA

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler

def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(),
                        nn.Linear(512,  c_out))

def subnet_conv(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 256,   3, padding=1), nn.ReLU(),
                        nn.Conv2d(256,  c_out, 3, padding=1))

def subnet_conv_1x1(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 256,   1), nn.ReLU(),
                        nn.Conv2d(256,  c_out, 1))

def generate_data(n_means, n_sample, x_dim, radius, std, show_figure=False):
    # Make data
    X_raw = np.zeros((n_means, n_sample, x_dim), dtype='float32')
    for i in range(n_means):
        th = 2*np.pi / n_means * (i+1)
        mean = [radius*np.cos(th), radius*np.sin(th)]
        X_raw[i, :, :] = np.random.multivariate_normal(mean, np.identity(x_dim)*std, size=n_sample)

    if show_figure:
        fig, ax = plt.subplots(figsize=(5,5), facecolor='white')
        for i in range(n_means):
            ax.scatter(X_raw[i,:,0], X_raw[i,:,1], s=1)
        print(X_raw.shape)
    return X_raw

def preprocess_data(n_sample, X_raw, x_dim, labels, show_figure=False):
    uq_labels = list(set(labels))
    idx2lab = {i:lab for i, lab in enumerate(uq_labels)}
    lab2idx = {idx2lab[key]:i for i, key in enumerate(idx2lab.keys())}

    X = X_raw.reshape((-1, x_dim))
    X = StandardScaler().fit_transform(X)
    y = [[lab2idx[lab]]*n_sample for lab in labels]
    y = list(itertools.chain.from_iterable(y)) # flatten
    y_onehot = np.eye(len(uq_labels))[y].astype('int')

    if show_figure:
        fig, ax = plt.subplots(figsize=(5,5), facecolor='white')
        for i, color in zip(idx2lab.keys(), lab2idx.keys()):
            idx = [True if j==i else False for j in y]
            ax.scatter(X[idx,0], X[idx,1], s=1, c=color)

        plt.show()

    return X, y_onehot

def batch_generator(data, labels, batch_size=16):
    assert data.shape[0] == labels.shape[0], "initial dimension of data and labels need to be the same"
    for idx in range(0, data.shape[0], batch_size):
        if idx + batch_size >= data.shape[0]:
            yield data[idx:-1], labels[idx:-1]
            break
        yield data[idx:idx+batch_size], labels[idx:idx+batch_size]

def main():
    # Config 
    n_means = 8
    radius = 14
    sd = 1
    labels = ['red','gold','green','chocolate','blue','magenta','pink','purple']
    # labels = ['red','red','red','red','blue','blue','green','purple']
    assert len(labels) == n_means

    x_dim = 2
    y_dim = len(list(set(labels)))
    z_dim = 2
    tot_dim = y_dim + z_dim
    pad_dim = tot_dim - x_dim
    n_sample = 256
    n_data = n_sample * n_means
    n_couple_layer = 3
    n_hid_layer = 3
    n_hid_dim = 512

    n_batch = 32
    n_epoch = 1000
    n_display = 100
    ## 

    X_raw = generate_data(n_means, n_sample, x_dim, radius, sd)

    X, y_onehot = preprocess_data(n_sample, X_raw, x_dim, labels)


    # Preprocess

    # Pad data
    pad_x = np.zeros((X.shape[0], pad_dim))
    x_data = np.concatenate([X, pad_x], axis=-1).astype('float32')
    z = np.random.multivariate_normal([0.]*x_dim, np.eye(x_dim), X.shape[0])
    y_data = np.concatenate([z, y_onehot], axis=-1).astype('float32')

    print(x_data.shape)
    print(y_data.shape)

    # # Make dataset generator
    # x_data = tf.data.Dataset.from_tensor_slices(x_data)
    # y_data = tf.data.Dataset.from_tensor_slices(y_data)
    # dataset = (tf.data.Dataset.zip((x_data, y_data))
    #         .shuffle(buffer_size=X.shape[0])
    #         .batch(n_batch, drop_remainder=True)
    #         .repeat())

    # Compiling
    inn = Ff.SequenceINN(x_dim)
    for k in range(8):
        inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)

    # Training
    optimizer = torch.optim.Adam(inn.parameters(), lr=0.001)

    # a very basic training loop
    MSE_loss = nn.MSELoss()
    for data, label in batch_generator(x_data, y_data, batch_size=n_batch):
        optimizer.zero_grad()

        x = torch.Tensor(data)
        # pass to INN and get transformed variable z and log Jacobian determinant
        z, log_jac_det = inn(x[:, 0:x_dim], c=[label[:, x_dim:tot_dim]])
        # z = zy[:, 0:z_dim]
        # y = zy[:, z_dim+1:tot_dim]
        # # calculate the negative log-likelihood of the model with a standard normal prior
        loss = 0.5*torch.sum(z**2, 1) - log_jac_det
        loss = loss.mean() / z_dim

        # # Calculate the y_loss
        # y_loss = MSE_loss(y, torch.Tensor(label[:, z_dim:tot_dim]))

        # # Combine the losses
        # loss = z_loss + y_loss

        # backpropagate and update the weights
        loss.backward()
        optimizer.step()

    # sample from the INN by sampling from a standard normal and transforming
    # it in the reverse direction
    z = torch.randn(n_data, z_dim)
    x_pred, _ = inn(z, c=[torch.Tensor(y_data[:, z_dim:tot_dim])], rev=True)
    print(x_pred.shape)
    x_pred = x_pred.detach().numpy()


    uq_labels = list(set(labels))
    idx2lab = {i:lab for i, lab in enumerate(uq_labels)}
    lab2idx = {idx2lab[key]:i for i, key in enumerate(idx2lab.keys())}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5), facecolor='white', sharex=True, sharey=True)
    for i, color in zip(idx2lab.keys(), lab2idx.keys()):
        idx = [True if j==i else False for j in y_onehot.argmax(axis=-1)]
        ax1.scatter(X[idx,0], X[idx,1], s=1, c=color)
        ax2.scatter(x_pred[idx,0], x_pred[idx,1], s=1, c=color)
        ax2.set_xlim([-2, 2])
        ax2.set_ylim([-2, 2])
    plt.suptitle('Original (left)                    Prediction (right)', fontsize=20);
    plt.show()


if __name__ == '__main__':
    main()
