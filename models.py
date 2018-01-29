import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import to_gpu
import json
import os


class MLP_D(nn.Module):
    def __init__(self, ninput, noutput,ncategory, layers,
                 activation=nn.LeakyReLU(0.2), gpu=False):
        super(MLP_D, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        self.ncategory = ncategory

#         layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        layer_sizes = [int(x) for x in layers.split('-')]
        self.layers = []
        
        clayer1 = nn.Linear(ninput, layer_sizes[0])
        self.layers.append(clayer1)
        self.add_module("clayer1", clayer1)
        clayer1_bn = nn.BatchNorm1d(layer_sizes[0])
        self.layers.append(clayer1_bn)
        self.add_module("clayer1_bn", clayer1_bn)
        self.layers.append(activation)
        self.add_module("activation_c1", activation)
        clayer2 = nn.Linear(ncategory,layer_sizes[0])
        self.layers.append(clayer2)
        self.add_module("clayer2", clayer2)
        clayer2_bn = nn.BatchNorm1d(layer_sizes[0])
        self.layers.append(clayer2_bn)
        self.add_module("clayer2_bn", clayer2_bn)
        self.layers.append(activation)
        self.add_module("activation_c2", activation)
        
        for i in range(len(layer_sizes)-1):
            if i == 0:
                layer = nn.Linear(layer_sizes[i]*2, layer_sizes[i+1])
            else : 
                layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            # No batch normalization after first layer
            if i != 0:
                bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, input, label):
        x = self.layers[2](self.layers[1](self.layers[0](input)))
        y=  self.layers[5](self.layers[4](self.layers[3](Variable(label))))
        x= torch.cat([x,y],1)
        for i in range(6,len(self.layers)):
            x = self.layers[i](x)
        x = torch.mean(x)
        return x


    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

# class MLP_D(nn.Module):
#     def __init__(self, ninput, noutput, layers,
#                  activation=nn.LeakyReLU(0.2), gpu=False):
#         super(MLP_D, self).__init__()
#         self.ninput = ninput
#         self.noutput = noutput

#         layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
#         self.layers = []

#         for i in range(len(layer_sizes)-1):
#             layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
#             self.layers.append(layer)
#             self.add_module("layer"+str(i+1), layer)

#             # No batch normalization after first layer
#             if i != 0:
#                 bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
#                 self.layers.append(bn)
#                 self.add_module("bn"+str(i+1), bn)

#             self.layers.append(activation)
#             self.add_module("activation"+str(i+1), activation)

#         layer = nn.Linear(layer_sizes[-1], noutput)
#         self.layers.append(layer)
#         self.add_module("layer"+str(len(self.layers)), layer)

#         self.init_weights()

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#         x = torch.mean(x)
#         return x

#     def init_weights(self):
#         init_std = 0.02
#         for layer in self.layers:
#             try:
#                 layer.weight.data.normal_(0, init_std)
#                 layer.bias.data.fill_(0)
#             except:
#                 pass
            
class MLP_G(nn.Module):
    def __init__(self, ninput, noutput,ncategory, layers,
                 activation=nn.ReLU(), gpu=False):
        super(MLP_G, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        self.ncategory = ncategory 

#         layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        layer_sizes = [int(x) for x in layers.split('-')]
        self.layers = []
        
		#20128-01-22 cgan
#        self.fc1_1 = nn.Linear(ninput, 300)
#        self.fc1_1_bn = nn.BatchNorm1d(300)
#        self.fc1_2 = nn.Linear(nlabel, 300)
#        self.fc1_2_bn = nn.BatchNorm1d(300)
#        self.fc2 = nn.Linear(300, 300)
#        self.fc2_bn = nn.BatchNorm1d(300)
#        self.fc3 = nn.Linear(300, noutput)
#        self.fc3_bn = nn.BatchNorm1d(1024)
#        self.fc4 = nn.Linear(1024, 784)
		
		
        clayer1 = nn.Linear(ninput, layer_sizes[0])
        self.layers.append(clayer1)
        self.add_module("clayer1", clayer1)
        clayer1_bn = nn.BatchNorm1d(layer_sizes[0])
        self.layers.append(clayer1_bn)
        self.add_module("clayer1_bn", clayer1_bn)	
        self.layers.append(activation)
        self.add_module("activation_c1", activation)	
		
        clayer2 = nn.Linear(ncategory, layer_sizes[0])
        self.layers.append(clayer2)
        self.add_module("clayer2", clayer2)			
        clayer2_bn = nn.BatchNorm1d(layer_sizes[0])
        self.layers.append(clayer2_bn)
        self.add_module("clayer2_bn", clayer2_bn)	
        self.layers.append(activation)
        self.add_module("activation_c2", activation)	
		
        for i in range(len(layer_sizes)-1):
            if i ==0 :
                layer = nn.Linear(layer_sizes[i]*2, layer_sizes[i+1])
            else : 
                layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
            self.layers.append(bn)
            self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)
        # print("self.layers_test")
        #
        # print(self.layers)
        # print(len(self.layers))
        self.init_weights()
        
    def forward(self, input, label):
        x = self.layers[2](self.layers[1](self.layers[0](input)))
        y = self.layers[5](self.layers[4](self.layers[3](Variable(label))))
        x= torch.cat([x,y],1)
        for i in range(6,len(self.layers)):
            x = self.layers[i](x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

#class MLP_G(nn.Module):
#    def __init__(self, ninput, noutput, layers,
#                 activation=nn.ReLU(), gpu=False):
#        super(MLP_G, self).__init__()
#        self.ninput = ninput
#        self.noutput = noutput
#
#        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
#        self.layers = []
#		
#        for i in range(len(layer_sizes)-1):
#            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
#            self.layers.append(layer)
#            self.add_module("layer"+str(i+1), layer)
#
#            bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
#            self.layers.append(bn)
#            self.add_module("bn"+str(i+1), bn)
#
#            self.layers.append(activation)
#            self.add_module("activation"+str(i+1), activation)
#
#        layer = nn.Linear(layer_sizes[-1], noutput)
#        self.layers.append(layer)
#        self.add_module("layer"+str(len(self.layers)), layer)
#
#        self.init_weights()
#
#    def forward(self, x):
#        for i, layer in enumerate(self.layers):
#            x = layer(x)
#        return x
#    def init_weights(self):
#        init_std = 0.02
#        for layer in self.layers:
#            try:
#                layer.weight.data.normal_(0, init_std)
#                layer.bias.data.fill_(0)
#            except:
#                pass
		

class Seq2Seq(nn.Module):
    def __init__(self, emsize, nhidden,ntokens, nlayers, noise_radius=0.2,
                 hidden_init=False, dropout=0, gpu=False):
        super(Seq2Seq, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_radius = noise_radius
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu

        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))

        # Vocabulary embedding
        self.embedding = nn.Embedding(ntokens, emsize)
        self.embedding_decoder = nn.Embedding(ntokens, emsize)

        # RNN Encoder and Decoder
        self.encoder = nn.LSTM(input_size=emsize,
                               hidden_size=nhidden,
                               num_layers=nlayers,
                               dropout=dropout,
                               batch_first=True)

        decoder_input_size = emsize+nhidden
        self.decoder = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=nhidden,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)

        # Initialize Linear Transformation
        self.linear = nn.Linear(nhidden, ntokens)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder.weight.data.uniform_(-initrange, initrange)

        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, bsz):
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2))

    def init_state(self, bsz):
        zeros = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, indices, lengths, noise, encode_only=False):
        batch_size, maxlen = indices.size()

        hidden = self.encode(indices, lengths, noise)

        if encode_only:
            return hidden

        if hidden.requires_grad:
            hidden.register_hook(self.store_grad_norm)

        decoded = self.decode(hidden, batch_size, maxlen,
                              indices=indices, lengths=lengths)

        return decoded

    def encode(self, indices, lengths, noise):
    
        
        embeddings = self.embedding(indices)
        
        packed_embeddings = pack_padded_sequence(input=embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        # Encode
        packed_output, state = self.encoder(packed_embeddings)

        hidden, cell = state
        # batch_size x nhidden
        hidden = hidden[-1]  # get hidden state of last layer of encoder

        # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        norms = torch.norm(hidden, 2, 1)
        
        # For older versions of PyTorch use:
        #hidden = torch.div(hidden, norms.expand_as(hidden))
        # For newest version of PyTorch (as of 8/25) use this:
        # hidden = torch.div(hidden, norms.unsqueeze(1).expand_as(hidden))

        if noise and self.noise_radius > 0:
            gauss_noise = torch.normal(means=torch.zeros(hidden.size()),
                                       std=self.noise_radius)
            hidden = hidden + to_gpu(self.gpu, Variable(gauss_noise))

        return hidden

    def decode(self, hidden, batch_size, maxlen, indices=None, lengths=None):
        # batch x hidden
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))

        else:
            state = self.init_hidden(batch_size)

        embeddings = self.embedding_decoder(indices)
        augmented_embeddings = torch.cat([embeddings, all_hidden], 2)

        packed_embeddings = pack_padded_sequence(input=augmented_embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        packed_output, state = self.decoder(packed_embeddings, state)

        output, lengths = pad_packed_sequence(packed_output, batch_first=True)

        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded = self.linear(output.contiguous().view(-1, self.nhidden))

        decoded = decoded.view(batch_size, maxlen, self.ntokens)


        return decoded
    
    def decode_mean(self, hidden, batch_size, maxlen, indices=None, lengths=None):
        # batch x hidden
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)



        embeddings = self.embedding_decoder(indices)

        augmented_embeddings = torch.cat([embeddings, all_hidden], 2)

        lengths2 = [augmented_embeddings.size()[1]]

        mean_augmented_embeddings = torch.mean(augmented_embeddings,0,True) #code 평균
        print(mean_augmented_embeddings)
        packed_embeddings = pack_padded_sequence(input=mean_augmented_embeddings,
                                                 lengths=lengths2,
                                                 batch_first=True)
        #packed_embeddings = pack_padded_sequence(input=augmented_embeddings,
        #                                         lengths=lengths,
        #                                         batch_first=True)
        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(1))

        else:
            state = self.init_hidden(1)
       
        

        packed_output, state = self.decoder(packed_embeddings, state)
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)


        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded = self.linear(output.contiguous().view(-1, self.nhidden))

        decoded = decoded.view(1, maxlen, self.ntokens)


        return decoded

    def generate(self, hidden, maxlen, sample=True, temp=1.0):
        """Generate through decoder; no backprop"""

        batch_size = hidden.size(0)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        # <sos>
        self.start_symbols.data.resize_(batch_size, 1)
        self.start_symbols.data.fill_(1)

        embedding = self.embedding_decoder(self.start_symbols)
        inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        # unroll
        all_indices = []
        for i in range(maxlen):
            output, state = self.decoder(inputs, state)
            overvocab = self.linear(output.squeeze(1))

            if not sample:
                vals, indices = torch.max(overvocab, 1)
            else:
                # sampling
                probs = F.softmax(overvocab/temp)
                indices = torch.multinomial(probs, 1)

            all_indices.append(indices)

            embedding = self.embedding_decoder(indices)
            inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)
#             inputs = torch.cat([embedding.unsqueeze(1), hidden.unsqueeze(1)], 2) # kong

        max_indices = torch.cat(all_indices, 1)
#         max_indices = torch.stack(all_indices, 1) # kong

        return max_indices
    
    def generate_code(self, hidden, maxlen, sample=True, temp=1.0):
        """Generate through decoder; no backprop"""

        batch_size = hidden.size(0)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        # <sos>
        self.start_symbols.data.resize_(batch_size, 1)
        self.start_symbols.data.fill_(1)

        embedding = self.embedding_decoder(self.start_symbols)
        inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        # unroll
        all_indices = []
        for i in range(maxlen):
            output, state = self.decoder(inputs, state)
            overvocab = self.linear(output.squeeze(1))

            if not sample:
                vals, indices = torch.max(overvocab, 1)
            else:
                # sampling
                vals, indices = torch.max(overvocab, 1)
                #probs = F.softmax(overvocab/temp)
                #indices = torch.multinomial(probs, 1)

            all_indices.append(indices)

            embedding = self.embedding_decoder(indices)
            #inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)
            inputs = torch.cat([embedding.unsqueeze(1), hidden.unsqueeze(1)], 2)

        #max_indices = torch.cat(all_indices, 1)
        max_indices = torch.stack(all_indices, 1)

        return max_indices

def load_models(load_path):
    model_args = json.load(open("{}/args.json".format(load_path), "r"))
    word2idx = json.load(open("{}/vocab.json".format(load_path), "r"))
    idx2word = {v: k for k, v in word2idx.items()}

    autoencoder = Seq2Seq(emsize=model_args['emsize'],
                          nhidden=model_args['nhidden'],
                          ntokens=model_args['ntokens'],
                          nlayers=model_args['nlayers'],
                          hidden_init=model_args['hidden_init'])
    gan_gen = MLP_G(ninput=model_args['z_size'],
                    noutput=model_args['nhidden'],
                    layers=model_args['arch_g'])
    gan_disc = MLP_D(ninput=model_args['nhidden'],
                     noutput=1,
                     layers=model_args['arch_d'])

    print('Loading models from'+load_path)
    ae_path = os.path.join(load_path, "autoencoder_model.pt")
    gen_path = os.path.join(load_path, "gan_gen_model.pt")
    disc_path = os.path.join(load_path, "gan_disc_model.pt")

    autoencoder.load_state_dict(torch.load(ae_path))
    gan_gen.load_state_dict(torch.load(gen_path))
    gan_disc.load_state_dict(torch.load(disc_path))
    return model_args, idx2word, autoencoder, gan_gen, gan_disc

def load_models_decode(load_path, ntokens):

    word2idx = json.load(open("{}/vocab.json".format(load_path), "r"))
    idx2word = {v: k for k, v in word2idx.items()}

    autoencoder = Seq2Seq(emsize=300,
                          nhidden=300,
                          ntokens=ntokens,
                          nlayers=1,
                          hidden_init=False)
    gan_gen = MLP_G(ninput=100,
                    noutput=300,
                    layers='300-300')
    gan_disc = MLP_D(ninput=300,
                     noutput=1,
                     layers='300-300')
					 
    print('Loading models from'+load_path)
    ae_path = os.path.join(load_path, "autoencoder_model.pt")
    gen_path = os.path.join(load_path, "gan_gen_model.pt")
    disc_path = os.path.join(load_path, "gan_disc_model.pt")

    autoencoder.load_state_dict(torch.load(ae_path))
    gan_gen.load_state_dict(torch.load(gen_path))
    gan_disc.load_state_dict(torch.load(disc_path))
    
    return idx2word, autoencoder, gan_gen, gan_disc, word2idx
    #word2idx추가(kong) model_args->delete
	    
