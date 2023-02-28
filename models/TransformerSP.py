from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils.net_util import norm_col_init, weights_init, toFloatTensor
import scipy.sparse as sp
import numpy as np

from datasets.glove import Glove
from .model_io import ModelOutput
from utils import flag_parser
from .Graph_TF import TransformerEncoder

from collections import deque

args = flag_parser.parse_arguments()


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        #(d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt).tocoo()


class TRANSFORMER_SP(torch.nn.Module):

    def __init__(self, args):
        action_space = args.action_space
        hidden_state_sz = args.hidden_state_sz
        super(TRANSFORMER_SP, self).__init__()

        # get and normalize adjacency matrix.
        np.seterr(divide='ignore')
        A_raw = torch.load("./data/gcn/adjmat.dat")
        A = normalize_adj(A_raw).tocsr().toarray()
        self.A = torch.nn.Parameter(torch.Tensor(A))

        n = int(A.shape[0])
        self.n = n

        self.embed_action = nn.Linear(action_space, 10)
        
        
        self.TFencoder = TransformerEncoder(200,512,512,512,512,[101,512],512,1024,2,4,0,use_bias=True)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8,batch_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.sup_embedding = nn.Linear(401,512)
        self.embuffer = deque(maxlen=2)
        self.K_frame = 0
        

        for i in range(4):
            place_holder = torch.zeros(1, 1027).cuda()
            self.embuffer.append(place_holder)
    
        
        

        lstm_input_sz = 10 + n * 5 + 512
        self.hidden_state_sz = hidden_state_sz
        self.lstm = nn.LSTMCell(lstm_input_sz, hidden_state_sz)
        num_outputs = action_space
        self.critic_linear = nn.Linear(1024, 1)
        self.actor_linear = nn.Linear(1024, num_outputs)

        

        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain("relu")
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0
        )
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.action_predict_linear = nn.Linear(2 * lstm_input_sz, action_space) # not used

        self.dropout = nn.Dropout(p=args.dropout_rate)

        # glove embeddings for all the objs.
        with open ("./data/gcn/objects.txt") as f:
            objects = f.readlines()
            self.objects = [o.strip() for o in objects]
        all_glove = torch.zeros(n, 300)
        glove = Glove(args.glove_file)
        for i in range(n):
            all_glove[i, :] = torch.Tensor(glove.glove_embeddings[self.objects[i]][:])

        self.all_glove = nn.Parameter(all_glove)
        self.all_glove.requires_grad = False

        self.W0 = nn.Linear(401, 401, bias=False)
        self.W1 = nn.Linear(401, 401, bias=False)
        self.W2 = nn.Linear(401, 5, bias=False)
        self.W3 = nn.Linear(10, 1, bias=False)

        self.final_mapping = nn.Linear(512, 1)
        self.final_mapping_t = nn.Linear(101, 512)

    def list_from_raw_obj(self, objbb, target):
        objstate = torch.zeros(self.n, 4)
        cos = torch.nn.CosineSimilarity(dim=1)
        glove_sim = cos(self.all_glove.detach(), target[None,:])[:,None]
        class_onehot = torch.zeros(1,self.n)
        for k, v in objbb.items():
            if k in self.objects:
                ind = self.objects.index(k)
            else:
                continue
            class_onehot[0][ind] = 1
            objstate[ind][0] = 1
            x1 = v[0::4]
            y1 = v[1::4]
            x2 = v[2::4]
            y2 = v[3::4]
            objstate[ind][1] = np.sum(x1+x2)/len(x1+x2) / 300
            objstate[ind][2] = np.sum(y1+y2)/len(y1+y2) / 300
            objstate[ind][3] = abs(max(x2) - min(x1)) * abs(max(y2) - min(y1)) / 300 / 300
        if args.gpu_ids != -1:
            objstate = objstate.cuda()
            class_onehot = class_onehot.cuda()
        objstate = torch.cat((objstate, glove_sim),dim=1)

        

        return objstate, class_onehot

    def new_gcn_embed(self, objstate, class_onehot):

        class_word_embed = torch.cat((class_onehot.repeat(self.n, 1), self.all_glove.detach()), dim=1) # (101,101+300) -> (101,401) 

        x = torch.mm(self.A, class_word_embed) 
        x = F.relu(self.W0(x)) # (101,401)

        x = x.unsqueeze(0) # (1,101,401)
        x = F.relu(self.sup_embedding(x)) # (1,101,512)
        x = self.TFencoder(x,None) # (1,101,512)


        """x = torch.mm(self.A, x) 
        x = F.relu(self.W1(x)) # (101,401)
        x = torch.mm(self.A, x) 
        x = F.relu(self.W2(x)) # (101,5)
        x = torch.cat((x, objstate), dim=1) # (101,5) -> (101,10)  
        x = torch.mm(self.A, x)
        x = F.relu(self.W3(x)) # (101,1) = self.last_mapping(x) # (1,512)"""

        print("x is now in the shape {}".format(x.shape))

        x = x.squeeze(0) # (101,512)
        x = F.relu(self.final_mapping(x)) # (101,1)
        x = x.view(1, self.n) # (1,101)


        x = self.final_mapping_t(x)

        

        return x

    def embedding(self, state, target, action_probs, objbb):
        state = state[None,:,:,:]
        objstate, class_onehot = self.list_from_raw_obj(objbb, target)
        action_embedding_input = action_probs
        action_embedding = F.relu(self.embed_action(action_embedding_input))
        x = objstate # (101,5)
        x = x.view(1, -1) #  (1,505)
        x = torch.cat((x, action_embedding), dim=1) # (1,515)
        out = torch.cat((x, self.new_gcn_embed(objstate, class_onehot)), dim=1) # (1,N*5+10+512) -> (1,1027)


        return out, None

    def a3clstm(self, embedding, prev_hidden): # embedding :(1,1027) 

        hx, cx = self.lstm(embedding, prev_hidden)
        x = hx
        actor_out = self.actor_linear(x)
        critic_out = self.critic_linear(x)
        return actor_out, critic_out, (hx, cx)


    def forward(self, model_input, model_options):

        state = model_input.state
        
        objbb = model_input.objbb
        (hx, cx) = model_input.hidden

        target = model_input.target_class_embedding
        action_probs = model_input.action_probs
        x, image_embedding = self.embedding(state, target, action_probs, objbb)

        actor_out, critic_out, (hx, cx) = self.a3clstm(x, (hx, cx))


        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
            embedding=image_embedding,
        )