# coding: utf-8

# In[2]:


#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
import torch as th
import torch.nn as nn
import numpy as np
import logging
import argparse
from torch.autograd import Variable
from collections import defaultdict as ddict
import torch.multiprocessing as mp
import model, train, rsgd
from model import HyperboloidDistance
from data import slurp
from rsgd import RiemannianSGD
from sklearn.metrics import average_precision_score
import gc
import sys
import time
import pickle

import pandas as pd
import numpy as np
import os

import global_variables
# th.set_default_tensor_type("torch.cuda.FloatTensor")

# In[3]:


def load_and_preprocess(filepath):
    '''
    Reads in the Amazon data, creates a unique index for each user and item, and filters out interactions
    with a rating of 1
    '''
    ratings_data = pd.read_csv(filepath, names=['user','item','rating','timestamp'])
    # remove interactions with rating = 1
    ratings_data = ratings_data[ratings_data['rating']>1]
    # map users and items to unique index
    all_users = ratings_data['user'].unique().tolist()
    all_items = ratings_data['item'].unique().tolist()
    all_users_items = all_users + all_items
    user_item_id_to_idx = {x:i for i, x in enumerate(all_users_items)}
    all_user_ids = [user_item_id_to_idx[x] for x in all_users]
    all_item_ids = [user_item_id_to_idx[x] for x in all_items]
    ratings_data['user'] = ratings_data['user'].apply(lambda x: user_item_id_to_idx[x])
    ratings_data['item'] = ratings_data['item'].apply(lambda x: user_item_id_to_idx[x])
    return ratings_data, user_item_id_to_idx, all_user_ids, all_item_ids

def get_most_recent_interaction_by_user(df):
    '''
    For each user, get their most recent interaction.
    If there are multiple interactions with the same timestamp, one interaction is selected randomly
    '''
    indices = df.groupby('user')['timestamp'].transform(max) == df['timestamp']
    recent_data = df[indices]
    recent_data = recent_data.groupby('user').apply(lambda x: x.sample(1))
    recent_data_original_indices = [x[1] for x in recent_data.index.tolist()]
    remaining_data = df.drop(index=recent_data_original_indices)
    return recent_data, remaining_data

def remove_test_items_not_seen_in_training(test_df, val_df, train_df):
    all_train_items = train_df.item.unique()
    test_df = test_df[test_df['item'].isin(all_train_items)]
    val_df = val_df[val_df['item'].isin(all_train_items)]
    return test_df, val_df

def train_val_test_split(df):
    '''
    Most recent interaction is saved for test and penultimate is saved for validation
    '''
    test_df, train_val_df = get_most_recent_interaction_by_user(df)
    val_df, train_df = get_most_recent_interaction_by_user(train_val_df)

    test_df,val_df = remove_test_items_not_seen_in_training(test_df, val_df, train_df)

    return train_df, val_df, test_df

def ndcg(top_n_hits):
    if sum(top_n_hits) == 0:
        return 0
    else:
        return 1/np.log2(2+list(top_n_hits).index(1))

# In[4]:

datafiles = os.listdir('amazon_data')
datafiles.sort()
for i, d in enumerate(datafiles):
    print(i, d)
# In[5]:


# select file

# datafile = datafiles[7]
datafile=datafiles[int(sys.argv[6])]

# datafile = datafiles[0]

# get data

input_data, user_item_id_to_idx, all_user_ids, all_item_ids = load_and_preprocess('amazon_data/' + datafile)

train_df, val_df, test_df = train_val_test_split(input_data)

train_df = train_df.sort_values('user')

pos_pairs = train_df.as_matrix(columns=['user', 'item'])

val_pos_pairs = val_df.as_matrix(columns=['user', 'item'])

# for each user create list of positive interactions that we will not sample from for our negative samples

user_pos_samples = train_df.groupby('user')['item'].apply(list).to_dict()

# In[6]:

# set model parameters

# n_dimensions = 100
# n_dimensions = 10
# learning_rate = 0.5
learning_rate=float(sys.argv[1])
# learning_rate = 0.001
# learning_rate = 0.001
# learning_rate = 0.05
# n_epochs = 20
# n_epochs = 100
# distance_function = 'poincare'
# distance_function = 'cosine'
# regularisation = 0.001
regularisation = float(sys.argv[2])
# regularisation = 0.1

n_dimensions = int(sys.argv[3])
global_variables.initialize(n_dimensions)

distance_function = sys.argv[4]
run_name=sys.argv[5]

print ("learning_rate [%f] regularisation [%f]" % (learning_rate, regularisation))
# In[7]:

# setup Riemannian gradients for distances
retraction = rsgd.euclidean_retraction
if distance_function == "hyperboloid":
    distfn = model.HyperboloidDistance
    rgrad = rsgd.hyperboloid_grad
    retraction = rsgd.hyperboloid_retraction
elif distance_function == 'poincare':
    distfn = model.PoincareDistance
    rgrad = rsgd.poincare_grad
elif distance_function == 'euclidean':
    distfn = model.EuclideanDistance
    rgrad = rsgd.euclidean_grad
elif distance_function == 'cosine':
    distfn = model.CosineDistance
    rgrad = rsgd.euclidean_grad
elif distance_function == 'transe':
    distfn = model.TranseDistance
    rgrad = rsgd.euclidean_grad
else:
    raise ValueError('Unknown distance function {}'.format(opt.distfn))


# In[14]:


# define our model

import numpy as np
from numpy.random import choice, randint
import torch as th
from torch import nn
from torch.autograd import Function, Variable
from torch.utils.data import Dataset
from collections import defaultdict as ddict
import torch.nn.functional as F

class Embedding(nn.Module):
    '''
    Generic embedding class taken from Nickel and Kiela (2017). Define an embedding layer (nn.Embedding) of
    a given size, initiate weights and define a forward function, which tells pytorch how the model behaves
    in the forward direction (as opposed to back propagation). In this case forward looks up embeddings yof
    all items passed to it within the embedding layer and then passes these to _forward, which will be defined
    in another class that defines our specific embedding model.
    '''
    def __init__(self, size, dim, dist=model.PoincareDistance, max_norm=1):
        super(Embedding, self).__init__()
        self.dim = dim
        self.lt = nn.Embedding(
            size, dim,
            # max_norm=max_norm,
            # sparse=True,
            sparse=False,
            scale_grad_by_freq=False
        )
        self.dist = dist
        self.init_weights()

    def init_weights(self, scale=1e-4):
        scale=0.001
        self.lt.state_dict()['weight'].uniform_(-scale, scale)

        weights=self.lt.state_dict()['weight']

    def forward(self, inputs):
        e = self.lt(inputs)
        norms=HyperboloidDistance.minkowski_tensor_dot(e,e)
        fval = self._forward(e)
        return fval

    def embedding(self):
        return list(self.lt.parameters())[0].data.cpu().numpy()


class CF_SNEmbedding(Embedding):
    '''
    Collaborative filtering model using Poincare distance. The _forward function takes embeddings for the query
    item (in this case will always be a user), the positive example (item the user purchased) and then a series of
    negative items (for the case of Bayesian Pairwise Loss this will just be one negative item).
    It then computes the Poincare distance between the query user and the positive and negative items and returns
    these distances multiplied by a learnable scale factor beta (as used in Vinh et al 2018).
    It also returns the embeddings themselves to be used for calculating the regularisation term in the loss function
    '''
    def __init__(self, size, dim, dist=model.PoincareDistance, max_norm=1):
        super(CF_SNEmbedding, self).__init__(size, dim, dist, max_norm)
        self.dist=dist

    def _forward(self, e):
        o = e.narrow(1, 1, e.size(1) - 1)
        s = e.narrow(1, 0, 1).expand_as(o)

        dists = self.dist()(s, o).squeeze(-1)
        return dists, e

class bpr_loss(nn.Module):
    '''
    Bayesian Pairwise Loss, based on Rendle et al 2012, and implemented as described in Vinh et al 2018.
    Take the log of the logistic sigmoid of the difference in scores between the positive and negative examples
    and then add a regularisation term based on the weights of the embedding layer.
    '''
    def __init__(self, regularisation):
        super(bpr_loss, self).__init__()
        self.regularisation = regularisation

    def forward(self, scores, embeddings):
        differences = scores[:,1] - scores[:,0]
        sig_diff = -th.log(F.sigmoid(differences))
        reg = self.regularisation * th.sum(embeddings * embeddings).cuda(0)
        res=(sig_diff + reg).mean()
        return res

# In[15]:

def bprl_loss_error(scores):
    # FIXME reversing pos and neg items order with positive distance function
    differences = scores[:,1] - scores[:,0]
    sig_diff = -th.log(F.sigmoid(differences))
    res=(sig_diff).mean()
    return res

def bprl_loss_norm(embeddings):
    res=th.mean(embeddings * embeddings).cuda(0)
    return res

cf = CF_SNEmbedding(len(user_item_id_to_idx), n_dimensions, distfn).cuda(0)
# In[16]:

# define our optimizer. This is taken directly from Nickel and Kiela 2017

optimizer = RiemannianSGD(
    cf.parameters(),
    rgrad=rgrad,
    retraction=retraction,
    lr=learning_rate,
)


# In[17]:


bprl = bpr_loss(regularisation)

# In[18]:


# train the model. Each epoch compute new random negative examples. Split into batches and print loss after each epoch

reg_lam = th.FloatTensor([regularisation]).cuda(0)
n_epochs=10

curr_lr=learning_rate
for epoch in range(n_epochs):
    params=list(cf.parameters())

    epoch_loss_error = []
    epoch_loss_norm = []
    loss = None

    lr = learning_rate

    #Training set
    random_negatives = np.random.choice(all_item_ids, (pos_pairs.shape[0], 1))
    input_triplets = np.concatenate((pos_pairs, random_negatives), axis=1)
    pos_items=np.copy(input_triplets[:,1])
    neg_items=np.copy(input_triplets[:,2])
    input_triplets = th.LongTensor(input_triplets)

    tripletloader = th.utils.data.DataLoader(input_triplets, batch_size=128, shuffle=True)

    #Validation set
    val_random_negatives = np.random.choice(all_item_ids, (val_pos_pairs.shape[0], 1))
    val_input_triplets = np.concatenate((val_pos_pairs, val_random_negatives), axis=1)

    pos_items=np.copy(val_input_triplets[:,1])
    neg_items=np.copy(val_input_triplets[:,2])

    val_input_triplets = th.LongTensor(val_input_triplets)

    zzz=0
    updated_curr_lr=False
    for batch in tripletloader:
        batch = th.autograd.Variable(batch).cuda(0)
        preds, e = cf(batch)
        loss = bprl(preds, e)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(lr=curr_lr)
        if epoch % 10 == 0 and not updated_curr_lr:
            curr_lr=float(curr_lr)/float(epoch+1)
            updated_curr_lr=True
        tmp_loss_error=bprl_loss_error(preds)
        tmp_loss_norm=bprl_loss_norm(e)
        epoch_loss_error.append(tmp_loss_error.data.item())
        epoch_loss_norm.append(tmp_loss_norm.data.item())

    updated_curr_lr=False
    print ("Epoch %i lr[%.2e]" % (epoch,curr_lr))
    val_batch = th.autograd.Variable(val_input_triplets).cuda(0)
    preds, e = cf(val_batch)
    nan_in_embeddings=th.any(th.isnan(e)) | th.any(th.isinf(e))
    if nan_in_embeddings:
        print ("NANs in embeddings!")
        print (e)
        sys.exit(11)

    nan_in_distance_scores=th.any(th.isnan(preds)) | th.any(th.isinf(preds))
    if nan_in_distance_scores:
        print ("NANs in distance scores!")
        print (preds)
        sys.exit(12)

    loss = bprl(preds, e)
    loss_error=bprl_loss_error(preds)
    loss_norm=bprl_loss_norm(e)
    print ("loss_error_train=%f" % np.mean(epoch_loss_error))
    print ("loss_error_val=%f" % loss_error)
    print (np.mean(epoch_loss_norm))

    print(epoch, np.mean(epoch_loss_error), float(loss_error.data.item()), np.mean(epoch_loss_norm), float(loss_norm.data.item()))
    print (th.mean(preds[:,0]),th.mean(preds[:,1]))
    print (th.mean(preds[:,0]-preds[:,1]))

    input_batch=th.autograd.Variable(input_triplets).cuda(0)
    preds, e = cf(input_batch)
    print (th.mean(preds[:,0]),th.mean(preds[:,1]))
    print (th.mean(preds[:,0]-preds[:,1]))

    # break

# In[19]:

# create test samples. These consist of the user, their most recent purchase and 100 negative items that
# the user has never purchased. We then compute scores for all 101 items and then rank the items.

test_pairs = test_df.as_matrix(columns=['user', 'item'])

print (test_pairs.shape)
print ("grouping user and items ...")
startTime=time.time()
user_pos_samples = input_data.groupby('user')['item'].apply(list).to_dict()
endTime=time.time()
print ("groping user and items. Done. [%s]" % (endTime-startTime))

no_neg_samples = 100
# no_neg_samples = 1000

neg_samples = np.zeros((test_pairs.shape[0],no_neg_samples), dtype=int)

# sys.exit(3)
dataset_name=datafile.replace("ratings_","").replace("_preprocessed.csv", "").lower()
samples_path="%s_test_samples.pkl" % dataset_name
if not os.path.isfile(samples_path):
    print ("negative sampling ...")
    startTime=time.time()
    for i in range(test_pairs.shape[0]):
        user_id = test_pairs[i,0]
        neg_candidates = list(set(all_item_ids).symmetric_difference(set(user_pos_samples[user_id])))
        neg_sample = np.random.choice(neg_candidates, no_neg_samples)
        neg_samples[i,:] = neg_sample
    endTime=time.time()
    print ("megative sampling. Done. [%s]" % (endTime-startTime))
    test_samples = np.concatenate((test_pairs, neg_samples), axis=1)

    pickle.dump(test_samples, open(samples_path, "wb"))
else:
    print ("loading %s" % samples_path)
    test_samples=pickle.load(open(samples_path, "rb"))

pos_items=np.copy(test_samples[:,1])
neg_items=np.copy(test_samples[:,2])

print (test_samples.shape)
test_samples = th.autograd.Variable(th.LongTensor(test_samples))

filename = "model_%s_clothing_lr%f_reg%f_ndim%i_run%s.torch" % (distance_function, learning_rate, regularisation, n_dimensions, run_name)
print ("saving %s ..." % filename)
startTime=time.time()
th.save(cf, filename)
endTime=time.time()
print ("saving. Done. [%s]" % (endTime-startTime))

print ("loading model on CPU ...")
startTime=time.time()
cf=th.load(filename, map_location=lambda storage, location: storage)
endTime=time.time()
print ("loading Done. [%s]" % (endTime-startTime))


test_scores, _ = cf(test_samples)

print (th.mean(test_scores[:,0]),th.mean(test_scores[:,1]))
print (th.mean(test_scores[:,0]-test_scores[:,1]))

test_scores = test_scores.data.numpy()

test_ranks = np.argsort(test_scores, axis=1)

# In[20]:


# Computes hit rate at N - how often the purchased item appears in the top N of our ranked list of
# product scores

top_n = 10

hits = 0
degenerated = 0
print ("here %i" % test_ranks.shape[0])
print (test_ranks.shape)
for i in range(test_ranks.shape[0]):
    if 0 in test_ranks[i,:top_n]:
        # s=np.copy(-test_scores[i,])
        s=np.copy(test_scores[i,])
        s_shuffle=np.copy(s)
        s.sort()
        sub_s=np.copy(s[0:50])
        print (sub_s)
        sub_s_shuffle=np.copy(sub_s)
        np.random.shuffle(sub_s_shuffle)
        if not np.array_equal(s, s_shuffle) and not np.array_equal(sub_s, sub_s_shuffle):
            hits += 1
        else:
            degenerated += 1
print ("nb hits [%i] nb degenerated [%i]" % (hits, degenerated))
hit_rate = hits/test_ranks.shape[0]
print('HR@'+str(top_n)+':', hit_rate)


# In[21]:
# Computes Normalized Discounted Cumulative Gain at N

top_n = 10

test_ranks_topn = test_ranks[:,:top_n]

y_test = np.zeros(test_ranks_topn.shape)
y_test[np.where(test_ranks_topn==0)] = 1

ndcgs = []
for i in range(y_test.shape[0]):
    ndcgs.append(ndcg(y_test[i,:]))
print('NDCG@'+str(top_n)+':', np.mean(ndcgs))
