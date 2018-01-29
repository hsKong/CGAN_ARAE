
# coding: utf-8

### Review data cluster result Load ###

import pickle
import os

name = 'dataexample.xlsx'
dataname = os.path.splitext(name)[0]

data_path = "./%s/" %dataname
if not os.path.exists(data_path):
    os.mkdir(data_path)


#pkl load
pkl_file = open(data_path+'DBSCAN_tV_raw_refined_docdict.pkl', 'rb')
DBSCAN_tV_raw_refined_doc = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open(data_path+'DBSCAN_tV_raw_refineddict.pkl', 'rb')
DBSCAN_tV_raw_refined = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open(data_path+'DBSCAN_tV_parsed_refined_docdict.pkl', 'rb')
DBSCAN_tV_parsed_refined_doc = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open(data_path+'DBSCAN_tV_parsed_refineddict.pkl', 'rb')
DBSCAN_tV_parsed_refined = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open(data_path+'rawdata_dicdict.pkl', 'rb')
rawdata_dic = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open(data_path+'raw_tokendict.pkl', 'rb')
raw_token = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open(data_path+'parsed_tokendict.pkl', 'rb')
parsed_token = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open(data_path+'parsed_dicdict.pkl', 'rb')
parsed_dic = pickle.load(pkl_file)
pkl_file.close()

#noise delete
del DBSCAN_tV_raw_refined_doc['-1_1']




ncategory = len(DBSCAN_tV_raw_refined_doc.keys())
label = DBSCAN_tV_raw_refined_doc.keys()
labellist = list(label)




DBSCAN_tV_raw_refined_doclist=[]
for key, value in DBSCAN_tV_raw_refined_doc.items():
    for i in value : 
        temp = [i,key]
        DBSCAN_tV_raw_refined_doclist.append(temp)


# In[416]:

# DBSCAN_tV_raw_refined_doclist


# In[28]:

# for x_,y_ in DBSCAN_tV_raw_refined_doclist : 
#     print(x_,y_)


# In[690]:

# import importlib
# importlib.reload(models)


### Train model ###
# In[691]:

import argparse
import os
import time
import math
import numpy as np
import random
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils import to_gpu, Corpus, batchify, train_ngram_lm, get_ppl, batchify_cluster
from models import Seq2Seq, MLP_D, MLP_G


# In[692]:

# fix input (2017-11-28 kong)
# Path Arguments
# data_path = 'data_korean'
kenlm_path = '../Data/kenlm'
outf = 'outcgan0128'
vocab_size = 11000
maxlen = 100

# Model Arguments
emsize = 300
nhidden = 300
ncategory = ncategory
nlayers = 1
noise_radius = 0.2
noise_anneal = 0.995
arch_g='300-300'
arch_d='300-300'
z_size=100
temp=1
enc_grad_norm=True
gan_toenc=0.01
dropout=0.0

# Training Arguments
epochs = 20
min_epochs = 50
no_earlystopping=False
patience=5
batch_size=5
niters_ae=1
niters_gan_d=5
niters_gan_g=1
niters_gan_schedule='2-4-6'
lr_ae=1
lr_gan_g=5e-05
lr_gan_d=1e-05
beta1=0.9
clip=1
gan_clamp=0.01

# Evaluation Arguments
sample = True
N=5
log_interval=200
# Other
seed=1111
cuda = False

# make output directory if it doesn't already exist
if not os.path.isdir('./output'):
    os.makedirs('./output')
if not os.path.isdir('./output/{}'.format(outf)):
    os.makedirs('./output/{}'.format(outf))

# Set the random seed manually for reproducibility.
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    if not cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(seed)


# In[693]:

###############################################################################
# Load data
###############################################################################

# create corpus
corpus = Corpus(data_path,
                maxlen=maxlen,
                clusterlist = DBSCAN_tV_raw_refined_doclist,
                vocab_size=vocab_size,
                lowercase=False)
# dumping vocabulary
with open('./output/{}/vocab.json'.format(outf), 'w') as f:
    json.dump(corpus.dictionary.word2idx, f)

# save arguments
ntokens = len(corpus.dictionary.word2idx)
print("Vocabulary Size: {}".format(ntokens))
# args.ntokens = ntokens

eval_batch_size = 10
# test_data = batchify(corpus.test, eval_batch_size, shuffle=False)
# train_data = batchify(corpus.train, batch_size, shuffle=True)
test_data_cluster=batchify_cluster(corpus.train_list, batch_size, shuffle=True) # kong
train_data_cluster=batchify_cluster(corpus.train_list, batch_size, shuffle=True) # kong

print("Loaded data!")


# In[694]:

###############################################################################
# Build the models
###############################################################################

ntokens = len(corpus.dictionary.word2idx)
autoencoder = Seq2Seq(emsize=emsize,
                      nhidden=nhidden,
                      ntokens=ntokens,
                      nlayers=nlayers,
                      noise_radius=noise_radius,
                      hidden_init=False,
                      dropout=dropout,
                      gpu=cuda)


# In[695]:

gan_gen = MLP_G(ninput=z_size, noutput=nhidden, ncategory = ncategory, layers=arch_g)


# In[696]:

gan_disc = MLP_D(ninput=nhidden, noutput=1, ncategory = ncategory, layers=arch_d)


# In[697]:

#1204delete
#print(autoencoder)
#print(gan_gen)
#print(gan_disc)

optimizer_ae = optim.SGD(autoencoder.parameters(), lr=lr_ae)
optimizer_gan_g = optim.Adam(gan_gen.parameters(),
                             lr=lr_gan_g,
                             betas=(beta1, 0.999))
optimizer_gan_d = optim.Adam(gan_disc.parameters(),
                             lr=lr_gan_d,
                             betas=(beta1, 0.999))

criterion_ce = nn.CrossEntropyLoss()

if cuda:
    autoencoder = autoencoder.cuda()
    gan_gen = gan_gen.cuda()
    gan_disc = gan_disc.cuda()
    criterion_ce = criterion_ce.cuda()


# In[728]:

###############################################################################
# Training code
###############################################################################


def save_model():
    print("Saving models")
    with open('./output/{}/autoencoder_model.pt'.format(outf), 'wb') as f:
        torch.save(autoencoder.state_dict(), f)
    with open('./output/{}/gan_gen_model.pt'.format(outf), 'wb') as f:
        torch.save(gan_gen.state_dict(), f)
    with open('./output/{}/gan_disc_model.pt'.format(outf), 'wb') as f:
        torch.save(gan_disc.state_dict(), f)


def evaluate_autoencoder(data_source, epoch):
    # Turn on evaluation mode which disables dropout.
    autoencoder.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary.word2idx)
    all_accuracies = 0
    bcnt = 0
    for i, batch in enumerate(data_source):
        source, target, lengths, label = batch # kong
        source = to_gpu(cuda, Variable(source, volatile=True))
        target = to_gpu(cuda, Variable(target, volatile=True))

        mask = target.gt(0)
        masked_target = target.masked_select(mask)
        # examples x ntokens
        output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)

        # output: batch x seq_len x ntokens
        output = autoencoder(source, lengths, noise=True)
        flattened_output = output.view(-1, ntokens)

        masked_output =             flattened_output.masked_select(output_mask).view(-1, ntokens)
        total_loss += criterion_ce(masked_output/temp, masked_target).data

        # accuracy
        max_vals, max_indices = torch.max(masked_output, 1)
        all_accuracies +=             torch.mean(max_indices.eq(masked_target).float()).data[0]
        bcnt += 1

        aeoutf = "./output/%s/%d_autoencoder.txt" % (outf, epoch)
        with open(aeoutf, "a") as f:
            max_values, max_indices = torch.max(output, 2)
            max_indices =                max_indices.view(output.size(0), -1).data.cpu().numpy()
            target = target.view(output.size(0), -1).data.cpu().numpy()
            for t, idx in zip(target, max_indices):
                # real sentence
                chars = " ".join([corpus.dictionary.idx2word[x] for x in t])
                f.write(chars)
                f.write("\n")
                # autoencoder output sentence
                chars = " ".join([corpus.dictionary.idx2word[x] for x in idx])
                f.write(chars)
                f.write("\n\n")

    return total_loss[0] / len(data_source), all_accuracies/bcnt

# kong
def evaluate_generator(noise, label, epoch):
    gan_gen.eval()
    autoencoder.eval()

    # generate from fixed random noise
    fake_hidden = gan_gen(noise, label)

    max_indices =         autoencoder.generate(fake_hidden, maxlen, sample=sample)
    
    with open("./output/%s/%s_generated.txt" % (outf, epoch), "w") as f:
        max_indices = max_indices.data.cpu().numpy()
#         print(max_indices)
        i = 0
        for idx in max_indices:
            # generated sentence
#             print(idx)
            words = [corpus.dictionary.idx2word[x] for x in idx]
            
            # truncate sentences to first occurrence of <eos>
            truncated_sent = []
            for w in words:
                if w != '<eos>':
                    truncated_sent.append(w)
                else:
                    break
            chars = " ".join(truncated_sent)
            f.write(str(fixed_y_.int().tolist()[i]))
            f.write(", ")
            f.write(chars)
            f.write("\n")
            i+=1
# def evaluate_generator(noise, epoch):
#     gan_gen.eval()
#     autoencoder.eval()

#     # generate from fixed random noise
#     fake_hidden = gan_gen(noise)
#     max_indices = \
#         autoencoder.generate(fake_hidden, maxlen, sample=sample)
#     with open("./output/%s/%s_generated.txt" % (outf, epoch), "w") as f:
#         max_indices = max_indices.data.cpu().numpy()
#         for idx in max_indices:
#             # generated sentence
#             words = [corpus.dictionary.idx2word[x] for x in idx]
#             # truncate sentences to first occurrence of <eos>
#             truncated_sent = []
#             for w in words:
#                 if w != '<eos>':
#                     truncated_sent.append(w)
#                 else:
#                     break
#             chars = " ".join(truncated_sent)
#             f.write(chars)
#             f.write("\n")


def train_lm(eval_path, save_path):
    # generate examples
    indices = []
    noise = to_gpu(cuda, Variable(torch.ones(100, z_size)))
    for i in range(1000):
        noise.data.normal_(0, 1)

        fake_hidden = gan_gen(noise)
        max_indices = autoencoder.generate(fake_hidden, maxlen)
        indices.append(max_indices.data.cpu().numpy())

    indices = np.concatenate(indices, axis=0)

    # write generated sentences to text file
#1204delete	
#    with open(save_path+".txt", "w") as f:
#        # laplacian smoothing
#        for word in corpus.dictionary.word2idx.keys():
#            f.write(word+"\n")
#        for idx in indices:
#            # generated sentence
#            words = [corpus.dictionary.idx2word[x] for x in idx]
#            # truncate sentences to first occurrence of <eos>
#            truncated_sent = []
#            for w in words:
#                if w != '<eos>':
#                    truncated_sent.append(w)
#                else:
#                    break
#            chars = " ".join(truncated_sent)
#            f.write(chars+"\n")

    # train language model on generated examples
#    lm = train_ngram_lm(kenlm_path=kenlm_path,
#                        data_path=save_path+".txt",
#                        output_path=save_path+".arpa",
#                        N=N)

    # load sentences to evaluate on
    with open(eval_path, 'r') as f:
        lines = f.readlines()
    sentences = [l.replace('\n', '') for l in lines]
    ppl = get_ppl(lm, sentences)

    return ppl


def train_ae(batch, total_loss_ae, start_time, i):
    autoencoder.train()
    autoencoder.zero_grad()

#     source, target, lengths = batch
    source, target, lengths, label = batch # kong
    source = to_gpu(cuda, Variable(source))
    target = to_gpu(cuda, Variable(target))


    # Create sentence length mask over padding
    mask = target.gt(0)
    masked_target = target.masked_select(mask)
    # examples x ntokens
    output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)

    # output: batch x seq_len x ntokens
    output = autoencoder(source, lengths, noise=True)

    # output_size: batch_size, maxlen, self.ntokens
    flattened_output = output.view(-1, ntokens)

    masked_output = flattened_output.masked_select(output_mask).view(-1, ntokens)
    loss = criterion_ce(masked_output/temp, masked_target)
    loss.backward()

    # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(autoencoder.parameters(), clip)
    optimizer_ae.step()

    total_loss_ae += loss.data

    accuracy = None
    if i % log_interval == 0 and i > 0:
        # accuracy
        probs = F.softmax(masked_output)
        max_vals, max_indices = torch.max(probs, 1)
        accuracy = torch.mean(max_indices.eq(masked_target).float()).data[0]

        cur_loss = total_loss_ae[0] / log_interval
        elapsed = time.time() - start_time
        print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
              'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}'
              .format(epoch, i, len(train_data_cluster), # kong
                      elapsed * 1000 / log_interval,
                      cur_loss, math.exp(cur_loss), accuracy))

        with open("./output/{}/logs.txt".format(outf), 'a') as f:
            f.write('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                   'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}\n'.
                   format(epoch, i, len(train_data_cluster),
                          elapsed * 1000 / log_interval,
                          cur_loss, math.exp(cur_loss), accuracy))

        total_loss_ae = 0
        start_time = time.time()

    return total_loss_ae, start_time


def train_gan_g(batch): # kong
    gan_gen.train()
    gan_gen.zero_grad()
    
    source, target, lengths, label = batch # kong
    source = to_gpu(cuda, Variable(source)) # kong
    target = to_gpu(cuda, Variable(target)) # kong
   
    y_label_ = torch.zeros(batch_size, ncategory) #kong
    y_label_.scatter_(1, label.view(batch_size, 1), 1) #kong
    
    noise = to_gpu(cuda,
                   Variable(torch.ones(batch_size,z_size)))
    noise.data.normal_(0, 1)

    fake_hidden = gan_gen(noise, y_label_) # kong
    errG = gan_disc(fake_hidden , y_label_) # kong

    # loss / backprop
    errG.backward(one)
    optimizer_gan_g.step()

    return errG


def grad_hook(grad):
    # Gradient norm: regularize to be same
    # code_grad_gan * code_grad_ae / norm(code_grad_gan)
    if enc_grad_norm:
        gan_norm = torch.norm(grad, 2, 1).detach().data.mean()
        normed_grad = grad * autoencoder.grad_norm / gan_norm
    else:
        normed_grad = grad

    # weight factor and sign flip
    normed_grad *= -math.fabs(gan_toenc)
    return normed_grad


def train_gan_d(batch):
    # clamp parameters to a cube
    for p in gan_disc.parameters():
        p.data.clamp_(-gan_clamp, gan_clamp)

    autoencoder.train()
    autoencoder.zero_grad()
    gan_disc.train()
    gan_disc.zero_grad()

    # positive samples ----------------------------
    # generate real codes
#     source, target, lengths = batch
    source, target, lengths, label = batch # kong
    source = to_gpu(cuda, Variable(source))
    target = to_gpu(cuda, Variable(target))


    y_label_ = torch.zeros(batch_size, ncategory) #kong
    y_label_.scatter_(1, label.view(batch_size, 1), 1) #kong

    # batch_size x nhidden
    real_hidden = autoencoder(source, lengths, noise=False, encode_only=True)
    real_hidden.register_hook(grad_hook)

    # loss / backprop
    errD_real = gan_disc(real_hidden, y_label_) # kong
    errD_real.backward(one)

    # negative samples ----------------------------
    # generate fake codes
    noise = to_gpu(cuda,
                   Variable(torch.ones(batch_size, z_size)))
    noise.data.normal_(0, 1)

    # loss / backprop
    fake_hidden = gan_gen(noise, y_label_) # kong
    errD_fake = gan_disc(fake_hidden.detach(), y_label_) #kong
    errD_fake.backward(mone)

    # `clip_grad_norm` to prvent exploding gradient problem in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(autoencoder.parameters(), clip)

    optimizer_gan_d.step()
    optimizer_ae.step()
    errD = -(errD_real - errD_fake)

    return errD, errD_real, errD_fake


# In[729]:

print("Training...")

with open("./output/{}/logs.txt".format(outf), 'a') as f:
    f.write('Training...\n')

# schedule of increasing GAN training loops
if niters_gan_schedule != "":
    gan_schedule = [int(x) for x in niters_gan_schedule.split("-")]
else:
    gan_schedule = []
niter_gan = 1

fixed_noise = to_gpu(cuda,
                     Variable(torch.ones(batch_size, z_size)))
fixed_noise.data.normal_(0, 1)
one = to_gpu(cuda, torch.FloatTensor([1]))
mone = one * -1

best_ppl = None
impatience = 0
all_ppl = []

# kong (for evaluation generator)
temp_z_ = fixed_noise
fixed_z_ = temp_z_
fixed_y_ = torch.zeros(batch_size, 1) + int(labellist[0])
# print(fixed_y_)
for i in labellist[1:]:
    
    fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
    temp_y = torch.zeros(batch_size,1) + int(i)
#     print(temp)
    fixed_y_ = torch.cat([fixed_y_, temp_y], 0)

# fixed_z_ = Variable(fixed_z_.cuda(), volatile=True)
fixed_y_label_ = torch.zeros(batch_size*ncategory, ncategory)
fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
# fixed_y_label_ = Variable(fixed_y_label_.cuda(), volatile=True)


for epoch in range(1, epochs+1):
    # update gan training schedule
    if epoch in gan_schedule:
        niter_gan += 1
        print("GAN training loop schedule increased to {}".format(niter_gan))
        
        with open("./output/{}/logs.txt".format(outf), 'a') as f:
            f.write("GAN training loop schedule increased to {}\n".
                   format(niter_gan))

    total_loss_ae = 0
    epoch_start_time = time.time()
    start_time = time.time()
    niter = 0
    niter_g = 0
    niter_d = 0
    niter_global = 1

    # loop through all batches in training data
    while niter < len(train_data_cluster): # kong

        # train autoencoder ----------------------------
        for i in range(niters_ae):
            if niter == len(train_data_cluster):  # kong
                break  # end of epoch
            total_loss_ae, start_time = train_ae(train_data_cluster[niter], total_loss_ae, start_time, niter) # kong
            niter += 1

        # train gan ----------------------------------
        for k in range(niter_gan):

            # train discriminator/critic
            for i in range(niters_gan_d):
                # feed a seen sample within this epoch; good for early training

                if niter_d == len(train_data_cluster):
                    errD, errD_real, errD_fake =                         train_gan_d(train_data_cluster[random.randint(0, len(train_data_cluster)-1)]) # kong
                else :
                    errD, errD_real, errD_fake =                         train_gan_d(train_data_cluster[niter_d]) # kong
                    niter_d += 1
                    
            # train generator
            for i in range(niters_gan_g):
                
                if niter_g == len(train_data_cluster ):
                    errG = train_gan_g(train_data_cluster[random.randint(0, len(train_data_cluster)-1)])
                else : 
                    errG = train_gan_g(train_data_cluster[niter_g])
                    niter_g+=1

        niter_global += 1
        if niter_global % 100 == 0:
            save_model()
            print('[%d/%d][%d/%d] Loss_D: %.8f (Loss_D_real: %.8f '
                  'Loss_D_fake: %.8f) Loss_G: %.8f'
                  % (epoch, epochs, niter, len(train_data_cluster), #kong
                     errD.data[0], errD_real.data[0],
                     errD_fake.data[0], errG.data[0]))
            
            with open("./output/{}/logs.txt".format(outf), 'a') as f:
                f.write('[%d/%d][%d/%d] Loss_D: %.8f (Loss_D_real: %.8f '
                       'Loss_D_fake: %.8f) Loss_G: %.8f\n'
                       % (epoch, epochs, niter, len(train_data_cluster), #kong
                          errD.data[0], errD_real.data[0],
                          errD_fake.data[0], errG.data[0]))
            print("------- %s seconds --------" % (time.time()-start))
            # exponentially decaying noise on autoencoder
            autoencoder.noise_radius =                 autoencoder.noise_radius*noise_anneal

            if niter_global % 3000 == 0:
                evaluate_generator(fixed_z_, fixed_y_label_, "epoch{}_step{}".format(epoch, niter_global))

#                # evaluate with lm
#                 if not no_earlystopping and epoch > min_epochs:
#                     ppl = train_lm(eval_path=os.path.join(data_path,"test.txt"),
#                                    save_path="output/{}/"
#                                             "epoch{}_step{}_lm_generations".
#                                             format(outf, epoch, niter_global))
#                     print("Perplexity {}".format(ppl))
#                     all_ppl.append(ppl)
#                     print(all_ppl)
#                     with open("./output/{}/logs.txt".format(outf), 'a') as f:
#                         f.write("\n\nPerplexity {}\n".format(ppl))
#                         f.write(str(all_ppl)+"\n\n")
#                     if best_ppl is None or ppl < best_ppl:
#                         impatience = 0
#                         best_ppl = ppl
#                         print("New best ppl {}\n".format(best_ppl))
                           
#                         with open("./output/{}/logs.txt".format(outf), 'a') as f:
#                             f.write("New best ppl {}\n".format(best_ppl))
#                         save_model()
#                     else:
#                         impatience += 1
#                         # end training
#                         if impatience > patience:
#                             print("Ending training")
                           
#                             with open("./output/{}/logs.txt".format(outf), 'a') as f:
#                                 f.write("\nEnding Training\n")
#                             sys.exit()
    # end of epoch ----------------------------
    # evaluation
    test_loss, accuracy = evaluate_autoencoder(test_data_cluster, epoch) # kong
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
          'test ppl {:5.2f} | acc {:3.3f}'.
          format(epoch, (time.time() - epoch_start_time),
                 test_loss, math.exp(test_loss), accuracy))
    print('-' * 89)
    with open("./output/{}/logs.txt".format(outf), 'a') as f:
        f.write('-' * 89)
        f.write('\n| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} |'
              ' test ppl {:5.2f} | acc {:3.3f}\n'.
              format(epoch, (time.time() - epoch_start_time),
                     test_loss, math.exp(test_loss), accuracy))
        f.write('-' * 89)
        f.write('\n')

    evaluate_generator(fixed_z_, fixed_y_label_, "end_of_epoch_{}".format(epoch))
#     if not no_earlystopping and epoch >= min_epochs:
#         ppl = train_lm(eval_path=os.path.join(data_path, "test.txt"),
#                       save_path="./output/{}/end_of_epoch{}_lm_generations".
#                                 format(outf, epoch))
#         print("Perplexity {}".format(ppl))
#         all_ppl.append(ppl)
#         print(all_ppl)
        
#         with open("./output/{}/logs.txt".format(outf), 'a') as f:
#             f.write("\n\nPerplexity {}\n".format(ppl))
#             f.write(str(all_ppl)+"\n\n")
#         if best_ppl is None or ppl < best_ppl:
#             impatience = 0
#             best_ppl = ppl
#             print("New best ppl {}\n".format(best_ppl))
#             with open("./output/{}/logs.txt".format(outf), 'a') as f:
#                 f.write("New best ppl {}\n".format(best_ppl))
#             save_model()
#         else:
#             impatience += 1
#             # end training
#             if impatience > patience:
#                 print("Ending training")
#                 with open("./output/{}/logs.txt".format(outf), 'a') as f:
#                     f.write("\nEnding Training\n")
#                 sys.exit()

    # shuffle between epochs
    save_model()
    train_data_cluster = batchify_cluster(corpus.train_list, batch_size, shuffle=True) #kong
# print("------- %s seconds --------" % (time.time()-start))

