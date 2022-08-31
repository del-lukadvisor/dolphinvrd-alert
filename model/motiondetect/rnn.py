import torch.nn as nn
import torch
import torch.nn.functional as F

from sklearn.utils import shuffle
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=512, output_size=9):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.drop = nn.Dropout(p=0.5)
        self.num_layers = 3
        self.dropout = 0.5
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True, num_layers=self.num_layers, dropout=self.dropout)
        
        self.W_s1 = nn.Linear(hidden_layer_size, 350)
        self.W_s2 = nn.Linear(350, 30)
        self.fc_layer = nn.Linear(30*1*hidden_layer_size, 2000)
        self.label = nn.Linear(2000, output_size)
        
        self.init_weights()
    
    def attention_net(self, lstm_output):
        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)
        return attn_weight_matrix

    def forward(self, input_seq, batch_size=1):

        #input_seq = input_seq.permute(1, 0, 2)

        h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).cuda())
        c_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).cuda())

        output, (h_n, c_n) = self.lstm(input_seq, (h_0, c_0))
        attn_weight_matrix = self.attention_net(output)
        hidden_matrix = torch.bmm(attn_weight_matrix, output)
        
        fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2]))
        logits = self.label(fc_out)

        return logits

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
