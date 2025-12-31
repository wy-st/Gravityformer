import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class nconv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class nconv_2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nvwl->ncwl', (x, A.permute(0, 2, 3, 1)))
        return x.contiguous()


class sGravity(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(2)
        self.lamda1 = nn.Parameter(2 * torch.randn(1) - 1, requires_grad=True)
        self.lamda2 = nn.Parameter(2 * torch.randn(1) - 1, requires_grad=True)
        self.lamda3 = nn.Parameter(2 * torch.randn(1) - 1, requires_grad=True)
        self.G = nn.Parameter(2 * torch.randn(1) - 1, requires_grad=True)

    def forward(self, x, dist):
        B, _, N, T = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B*T, N, -1)
        x = F.softplus(self.avgpool(x)) + 1
        x1, x2 = x.chunk(2, dim=-1)
        lamda1 = F.softplus(self.lamda1)  # 使用softplus保证lamda值为正
        lamda2 = F.softplus(self.lamda2)
        lamda3 = F.softplus(self.lamda3)
        G = F.softplus(self.G)
        dots = torch.matmul(torch.pow(x1, lamda1), torch.pow(x2, lamda2).transpose(-1, -2))
        dist2 = torch.pow(dist, 2 * lamda3)
        inter_att = G * dots * dist2
        return F.softmax(inter_att.reshape(B, T, N, N), dim=-1)


class feature(nn.Module):
    def __init__(self, c_in, c_out):
        super(feature, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class Dual_axis(nn.Module):
    def __init__(self, device, channels, num_nodes, time_steps, d_n, d_v, d_t, heads, dropout, divided_hidden=4):
        super().__init__()
        self.num_heads = heads
        self.hidden = channels // divided_hidden
        self.head_dim = self.hidden // heads
        # axis N attn
        self.fc_nq = nn.Linear(channels, self.hidden)
        self.fc_nk = nn.Linear(channels, self.hidden)
        self.sg = nconv_2()

        # axis T attn
        self.fc_tq = nn.Linear(channels, self.hidden)
        self.fc_tk = nn.Linear(channels, self.hidden)
        self.tg = nconv_2()


        self.v =  nn.Conv2d(channels, self.hidden, 1)
        # Transformer v
        self.fc_v = nn.Conv2d(channels, 2*self.hidden, 1)
        self.proj = nn.Conv2d(2*self.hidden, channels, 1)

        # other parameters
        self.channels = channels
        self.d_n = self.hidden // heads
        self.d_v = self.hidden // heads
        self.d_t = self.hidden // heads
        self.heads = heads
        self.dropout = dropout
        self.scaled_factor_n = self.head_dim ** -0.5
        self.scaled_factor_t = self.head_dim ** -0.5
        self.Bn = nn.Parameter(torch.Tensor(1, self.heads, num_nodes, num_nodes), requires_grad=True)
        self.Bt = nn.Parameter(torch.Tensor(1, self.heads, time_steps, time_steps), requires_grad=True)

    def forward(self, x, time_ind, weekin, holiin, adj_s):
        b, f, n, t = x.shape
        # N attn
        qk_n = x.permute(0, 3, 1, 2).reshape(b*t, f, n)
        nq = self.fc_nq(qk_n.squeeze().permute(0, 2, 1))  # [b*t, n, heads*d_n]
        nk = self.fc_nk(qk_n.squeeze().permute(0, 2, 1))
        nq = nq.view(-1, n, self.heads, self.d_n).permute(0, 2, 1, 3).contiguous()
        nk = nk.view(-1, n, self.heads, self.d_n).permute(0, 2, 1, 3).contiguous()
        attn_n = torch.einsum('... i d, ... j d -> ... i j', nq, nk) * self.scaled_factor_n
        attn_n = attn_n + self.Bn
        attn_n = torch.softmax(attn_n.reshape(b, t, self.heads, n, n), dim=-1) * adj_s.unsqueeze(2).repeat(1, 1, self.num_heads, 1, 1) # [b, heads, h, h] -> [3, 2, 112, 112]
        attn_n = attn_n.permute(0, 2, 1, 3, 4).reshape(b * self.heads, t, n, n)

        # T attn
        qk_t = x.permute(0, 2, 1, 3).reshape(b * n, f, t)
        tq = self.fc_tq(qk_t.squeeze().permute(0, 2, 1))   # [b, t, heads*d_n]
        tk = self.fc_tk(qk_t.squeeze().permute(0, 2, 1))
        tq = tq.view(-1, t, self.heads, self.d_t).permute(0, 2, 1, 3).contiguous()
        tk = tk.view(-1, t, self.heads, self.d_t).permute(0, 2, 1, 3).contiguous()
        attn_t = torch.einsum('... i d, ... j d -> ... i j', tq, tk) * self.scaled_factor_t
        attn_t = attn_t + self.Bt
        attn_t = torch.softmax(attn_t, dim=-1).reshape(b, n, self.heads, t, t)
        attn_t = attn_t.permute(0, 2, 1, 3, 4).reshape(b * self.heads, n, t, t)

        # gconv for spatial and temporal
        v1 = self.v(x)
        v1 = torch.cat(torch.split(v1, self.head_dim, dim=1), dim=0)
        sx = self.sg(v1, attn_n)
        hs = F.dropout(sx, self.dropout, training=self.training)
        xt = v1.transpose(2, 3).contiguous()
        tx = self.tg(xt, attn_t)
        ht = F.dropout(tx, self.dropout, training=self.training)
        ht = ht.transpose(2, 3).contiguous()
        h = torch.cat((ht, hs), dim=1)
        h = torch.cat(torch.split(h, b, dim=0), dim=1)
        v = self.fc_v(x)
        output = self.proj(h * v)
        return output


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dropout= 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * 4, (1, 1)),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * 2, dim, (1, 1)))

    def forward(self, x):
        return self.net(x)


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    def __init__(self, device, channels, num_nodes, time_steps, d_n, d_v, d_t, heads, dropout):
        super().__init__()
        self.norm1 = LayerNorm(channels, eps=1e-6)
        self.norm2 = LayerNorm(channels, eps=1e-6)
        self.att = Dual_axis(device, channels, num_nodes, time_steps, d_n, d_v, d_t, heads, dropout)
        self.mlp = FeedForward(channels, dropout)

    def forward(self, x, time_ind, weekin, holiin, adj_s):
        x = self.att(self.norm1(x), time_ind, weekin, holiin, adj_s) + x
        x = self.mlp(self.norm2(x)) + x
        return x


class HadamardMapper(nn.Module):
    def __init__(self, device, group, dim, learnable=True):
        super(HadamardMapper, self).__init__()
        self.device = device
        self.group = group
        self.dim = dim
        H_group = self.constructH(group)
        self.H = nn.Parameter(H_group.repeat(dim // group, 1, 1), requires_grad=learnable)

    def constructH(self, group):
        H = torch.ones(1, 1).to(self.device)
        for i in range(int(math.log2(group))):
            H = torch.cat((torch.cat([H, H], 1), torch.cat([H, -H], 1)), 0) / math.sqrt(2)
            if H.shape[0] == group:
                return H

    def forward(self, x):
        x_shape2 = x.shape
        x = x.reshape(-1, x.shape[-1])
        x = x.reshape(-1, self.dim // self.group, self.group).transpose(0, 1)
        x = torch.bmm(x, self.H).transpose(0, 1)
        x = x.reshape(x_shape2)
        return x


class G_Generactor(nn.Module):
    def __init__(self, device, num_nodes):
        super().__init__()
        self.device = device
        self.n = num_nodes
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 40), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(num_nodes, 40), requires_grad=True).to(device)

    def forward(self):
        nodevec1 = torch.tanh(2 * self.nodevec1)
        nodevec2 = torch.tanh(2 * self.nodevec2)
        a0 = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        a = F.relu(torch.tanh(2 * a0))
        return a


class GFormer(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3,
                 in_dim=1, out_dim=4, time_steps=12, residual_channels=32,
                 dilation_channels=32, skip_channels=256,
                 layers=6, adj=None):
        super().__init__()
        self.device = device
        self.dist = adj
        self.dropout = dropout
        self.layers = layers
        self.residual_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.result_fuse = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gravity = nn.ModuleList()

        self.layer_st = nn.ModuleList()
        self.fc1 = nn.ModuleList()
        self.fc2 = nn.ModuleList()
        for b in range(layers):
            self.gravity.append(sGravity())
            self.layer_st.append(Block(device=device, channels=residual_channels, num_nodes=num_nodes, time_steps=time_steps,
                                     d_n=residual_channels, d_v=residual_channels,
                                     d_t=residual_channels, heads=1, dropout=dropout))
            self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=residual_channels,
                                                 kernel_size=(1, 1)))

            self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                             out_channels=skip_channels,
                                             kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(residual_channels))
            self.result_fuse.append(torch.nn.Conv2d(dilation_channels * 1, residual_channels, kernel_size=(1, 1),
                                                    padding=(0, 0), stride=(1, 1), bias=True))

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=time_steps * residual_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.hm = HadamardMapper(device=device, group=4, dim=residual_channels)
        self.s_graph = G_Generactor(device, num_nodes)
        self.inflow_fc = nn.Sequential(nn.Conv2d(1, residual_channels, (1, 1)), nn.ReLU(),
                                       nn.Conv2d(residual_channels, residual_channels, (1, 1)))
        self.outflow_fc = nn.Sequential(nn.Conv2d(1, residual_channels, (1, 1)), nn.ReLU(),
                                        nn.Conv2d(residual_channels, residual_channels, (1, 1)))

        self.day_emb = nn.Embedding(48, dilation_channels)
        self.week_emb = nn.Embedding(7, dilation_channels)
        self.holi_emb = nn.Embedding(2, dilation_channels)
        self.adaptive_emb = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(dilation_channels * 2, num_nodes, time_steps)))
        self.transition = nn.Conv2d(in_channels=6 * dilation_channels,
                                    out_channels=dilation_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.nconv1 = nconv()
        self.nconv2 = nconv()
        self.to(device)

    def forward(self, overallinput, time_emb, time_wemb, time_hemb):
        time_ind = time_emb[:, 0, 0, -1]
        weekin = time_wemb[:, 0, 0, -1]
        holiin = time_hemb[:, 0, 0, -1]

        emb_day = self.day_emb(time_emb.squeeze()).permute(0, 3, 1, 2)
        emb_week = self.week_emb(time_wemb.squeeze()).permute(0, 3, 1, 2)
        emb_holi = self.holi_emb(time_hemb.squeeze()).permute(0, 3, 1, 2)
        emb_adaptive = self.adaptive_emb.unsqueeze(0).repeat(emb_day.size(0), 1, 1, 1)
        emb_feature = torch.cat((emb_day, emb_week, emb_holi, emb_adaptive), dim=1)

        input, inflow, outflow = overallinput[:, :, 0:1, :], overallinput[:, :, 1:2, :], overallinput[:, :, 2:3, :]
        x = input.permute(0, 2, 1, 3)
        x = self.start_conv(x)

        inflow = self.inflow_fc(inflow.permute(0, 2, 1, 3))
        outflow = self.outflow_fc(outflow.permute(0, 2, 1, 3))
        inflow = torch.cat((inflow, emb_feature), dim=1)
        outflow = torch.cat((outflow, emb_feature), dim=1)

        adj_s = self.s_graph()
        adj = self.dist * adj_s
        x = torch.cat((x, emb_feature), dim=1)
        x = self.transition(x)
        x = self.hm(x)

        skip = 0
        for i in range(self.layers):
            residual = x
            inflow1 = torch.cat((inflow, x), dim=1)
            outflow1 = torch.cat((outflow, x), dim=1)
            sgravity = self.gravity[i](torch.cat((inflow1, outflow1), dim=1), adj)
            st_feature = self.layer_st[i](x, time_ind, weekin, holiin, sgravity)
            x = F.relu(st_feature)
            x = self.result_fuse[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = torch.transpose(x, 3, 2)
        x = torch.reshape(x, (x.size(0), x.size(1) * x.size(2), x.size(3), 1))
        x = self.end_conv_2(x)
        return x.squeeze(-1).permute(0, 2, 1)

def make_model(DEVICE, input_size, hidden_size, num_of_vertices, num_of_timesteps, adj):
    '''

    :param DEVICE:
    :param nb_block:
    :param in_channels:
    :param K:
    :param nb_chev_filter:
    :param nb_time_filter:
    :param time_strides:
    :param cheb_polynomials:
    :param nb_predict_step:
    :param len_input
    :return:
    '''
    model = GFormer(DEVICE, num_of_vertices, dropout=0.3, in_dim=input_size, out_dim=num_of_timesteps, adj=adj)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model

if __name__ == '__main__':
    x = torch.FloatTensor(4, 170, 3, 12)
    dist = torch.FloatTensor(170, 170)
    time = torch.ones(4, 170, 1, 12).type(torch.LongTensor)
    wtime = torch.ones(4, 170, 1, 12).type(torch.LongTensor)
    htime = torch.ones(4, 170, 1, 12).type(torch.LongTensor)
    model = GFormer('cpu', 170, adj=dist)
    y = model(x, time, wtime, htime)
    print(y.shape)
