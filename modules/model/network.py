# Network
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
import torch.nn.functional as F
import math

def convbn(in_planes, out_planes, kernel_size, stride, pad=0, dilation=1, bias = False):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=bias), nn.BatchNorm2d(out_planes))

def conv(in_planes, out_planes, kernel_size, stride, pad=0, dilation=1, bias = False):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=bias))

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, pe = False, ln=False, attention_dropout = 0.1, dim_feedforward = 2048):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.num_heads = num_heads
        self.pe = pe

        # Projection
        self.fc_q = nn.Linear(dim_Q, dim_V) # dimin -> dimhidden
        self.fc_k = nn.Linear(dim_K, dim_V) # dimin -> dimhidden
        self.fc_v = nn.Linear(dim_K, dim_V) # dimhidden -> dim
        if pe:
            self.fc_p = nn.Linear(3, dim_V) # (lx, ly) -> (lx', ly')
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.dropout_attention = nn.Dropout(attention_dropout)
        self.fc_o1 = nn.Linear(dim_V, dim_feedforward)
        self.fc_o2 = nn.Linear(dim_feedforward, dim_V)
        self.dropout1 = nn.Dropout(attention_dropout)
        self.dropout2 = nn.Dropout(attention_dropout)

    def forward(self, Q, K, p = None):

        Q = self.fc_q(Q) # input_dim -> embed dim
        K, V = self.fc_k(K), self.fc_v(K) # input_dim -> embed dim
        dim_split = self.dim_V // self.num_heads # for multi-head attention
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        if self.pe:
            P = self.fc_p(p)
            P_ = torch.cat(P.split(dim_split, 2), 0)
            A = self.dropout_attention(torch.softmax(Q_.bmm(K_.transpose(1,2)) + P_.bmm(P_.transpose(1,2))/math.sqrt(self.dim_V), 2)) # Attention Dropout
        else:
            A = self.dropout_attention(torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)) # Attention Dropout

        A =  A.bmm(V_) # A(Q, K, V) attention_output
        O = torch.cat((Q_ + A).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + self.dropout2(self.fc_o2(self.dropout1(F.gelu(self.fc_o1(O)))))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False, attention_dropout = 0.1, pe = False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln, attention_dropout = attention_dropout, pe = pe)
        self.pe = pe
    def forward(self, X):
        if self.pe:
            x = X[:, :, :3] # observation
            p = X[:, :, 3:] # light
            return self.mab(x,x,p)
        else:
            return self.mab(X,X)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class Transformer(nn.Module):
    def __init__(self, dim_input, dim_output, num_enc_sab = 3, num_dec_sab = 0, num_outputs = 1, num_inds=32, dim_hidden=512, num_heads=8, ln=False, attention_dropout=0):
        super().__init__()

        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden

        modules_enc = []
        modules_enc.append(SAB(dim_input, dim_hidden, num_heads, ln=ln, attention_dropout = attention_dropout, pe=False))
        for k in range(num_enc_sab):
            modules_enc.append(SAB(dim_hidden, dim_hidden, num_heads, ln=ln, attention_dropout = attention_dropout, pe=False))
        self.enc = nn.Sequential(*modules_enc)
        modules_dec = []
        modules_dec.append(PMA(dim_hidden, num_heads, num_outputs)) # after the PMA we should not put drop out
        for k in range(num_dec_sab):
            modules_dec.append(SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        self.dec = nn.Sequential(*modules_dec)
        modules_regression = []
        modules_regression.append(nn.Linear(num_outputs * dim_hidden, dim_hidden))
        modules_regression.append(nn.ReLU())
        modules_regression.append(nn.Linear(dim_hidden, dim_output))
        self.regression = nn.Sequential(*modules_regression)

    def init_weights(self, zero = False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                if zero == False:
                    xavier_uniform_(m.weight)
                else:
                    xavier_uniform_(m.weight, gain = 1.0e-3)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        feat = x.view(-1, self.num_outputs * self.dim_hidden)
        x = self.regression(feat)
        return x, feat



class Encoder(nn.Module):
    def __init__(self, n_inputs):
        super(Encoder, self).__init__()

        self.first = nn.Sequential(conv(n_inputs, 32, kernel_size=3, stride=1, pad=1, dilation=1),
                                   nn.LeakyReLU(inplace=True),
                                   conv(32, 32, kernel_size=3, stride=1, pad=1, dilation=1),
                                   nn.LeakyReLU(inplace=True)
                                   )

        self.down1 = nn.Sequential(conv(32, 32, kernel_size=3, stride=1, pad=1, dilation=1),
                                   nn.LeakyReLU(inplace=True),
                                   conv(32, 64, kernel_size=3, stride=1, pad=1, dilation=1),
                                   nn.LeakyReLU(inplace=True))

        self.down2 = nn.Sequential(conv(64, 64, kernel_size=3, stride=1, pad=1, dilation=1),
                                   nn.LeakyReLU(inplace=True),
                                   conv(64, 64, kernel_size=3, stride=1, pad=1, dilation=1),
                                   nn.LeakyReLU(inplace=True))

    def init_weights(self, zero = False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if zero == False:
                    xavier_uniform_(m.weight)
                else:
                    xavier_uniform_(m.weight, gain = 1.0e-3)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        output1     = self.first(x)
        output2     = self.down1(output1)
        output3     = self.down2(output2)
        return output3

class DecoderRegressionFCN(nn.Module):
    def __init__(self, dims, n_outputs):
        super(DecoderRegressionFCN, self).__init__()
        self.first = nn.Sequential(convbn(dims, dims, kernel_size=3, stride=1, pad=1, dilation=1),
                                   nn.LeakyReLU(inplace=True),
                                   convbn(dims, dims, kernel_size=3, stride=1, pad=1, dilation=1),
                                   nn.LeakyReLU(inplace=True)
                                   )
        self.second = nn.Sequential(convbn(dims, dims, kernel_size=3, stride=1, pad=1, dilation=1),
                                   nn.LeakyReLU(inplace=True),
                                   convbn(dims, dims, kernel_size=3, stride=1, pad=1, dilation=1),
                                   nn.LeakyReLU(inplace=True)
                                   )
        self.predict = nn.Sequential(conv(dims, n_outputs, kernel_size=3, stride=1, pad=1, dilation=1))


    def init_weights(self, zero = False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if zero == False:
                    xavier_uniform_(m.weight)
                else:
                    xavier_uniform_(m.weight, gain = 1.0e-3)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        x = self.predict(x)
        return x
