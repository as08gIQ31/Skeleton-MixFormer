import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
#from .pos_embed import Pos_Embed
#from model.dropSke import DropBlock_Ske
#from model.dropT import DropBlockT_1d
from einops import rearrange
import thop
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

class Tem_Seq_h(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Tem_Seq_h, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)        
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)
        self.AvpTemSeq = nn.AdaptiveAvgPool2d(1)
        self.MaxTemSeq = nn.AdaptiveMaxPool2d(1)
        self.combine_conv = nn.Conv2d(2, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):    
        x = self.conv(x)
        N,C,T,V = x.size()
        x = x.permute(0,2,1,3).contiguous()
        Q_Tem_Seq = self.AvpTemSeq(x)
        K_Tem_Seq = self.MaxTemSeq(x)
        Combine = torch.cat([Q_Tem_Seq,K_Tem_Seq],dim=2)
        Combine = self.combine_conv(Combine.permute(0,2,1,3).contiguous()).permute(0,2,1,3).contiguous()
        Tem_Seq_out = (x * self.sigmoid(Combine).expand_as(x)).permute(0,2,1,3).contiguous()      
        return Tem_Seq_out
                             
class Tem_Trans(nn.Module):
    def __init__(self, in_channels, out_channels, Frames, kernel_size, stride=1):
        super(Tem_Trans, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))     
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.AvpTemTrans = nn.AdaptiveAvgPool2d(1)
        self.MaxTemTrans = nn.AdaptiveMaxPool2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()        
        self.soft = nn.Softmax(-1)
        self.linear = nn.Linear(Frames,Frames)
#        self.linear2 = nn.Linear(Frames,Frames)
    def forward(self, x):    
        x = self.conv(x)
        N,C,T,V=x.size()
        x1 = x[:,:C//2,:,:]
        x2 = x[:,C//2:C,:,:]                
        Q_Tem_Trans = self.AvpTemTrans(x1.permute(0,2,1,3).contiguous())
        K_Tem_Trans = self.MaxTemTrans(x2.permute(0,2,1,3).contiguous())
        Q_Tem_Trans = self.relu(self.linear(Q_Tem_Trans.squeeze(-1).squeeze(-1)))
        K_Tem_Trans = self.relu(self.linear(K_Tem_Trans.squeeze(-1).squeeze(-1)))       
        Tem_atten = self.sigmoid(torch.einsum('nt,nm->ntm', (Q_Tem_Trans, K_Tem_Trans)))                   
        Tem_Trans_out = self.bn(torch.einsum('nctv,ntm->ncmv', (x, Tem_atten)))      
        return Tem_Trans_out
        
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):               
        x = self.conv(x)
        x = self.bn(x)
        return x

class Temporal_MixFormer(nn.Module):
    def __init__(self,in_channels,out_channels,Frames,kernel_size=3,stride=1,dilations=[1,2,3,4],residual=True,residual_kernel_size=1):
        super().__init__()
        assert out_channels % (len(dilations) + 3) == 0
        self.num_branches = len(dilations) + 3
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        self.branches = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(in_channels,branch_channels,kernel_size=1,padding=0),nn.BatchNorm2d(branch_channels),nn.ReLU(inplace=True),
            TemporalConv(branch_channels,branch_channels,kernel_size=ks,stride=stride,dilation=dilation),)
            for ks, dilation in zip(kernel_size, dilations)
        ])
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)
        ))
        self.branches.append(nn.Sequential(
            Tem_Trans(in_channels, branch_channels, Frames, kernel_size=1, stride=stride),
            nn.BatchNorm2d(branch_channels)
        ))        
        self.branches.append(nn.Sequential(
            Tem_Seq_h(in_channels, branch_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(branch_channels)
        ))        
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)
        self.apply(weights_init)

    def forward(self, x):
        res = self.residual(x)        
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        out = torch.cat(branch_outs, dim=1)
        out += res
        return out

class Spa_Atten(nn.Module):
    def __init__(self, out_channels):
        super(Spa_Atten, self).__init__()
        self.out_channels=out_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(-1) 
        self.bn = nn.BatchNorm2d(out_channels)
        self.linear = nn.Linear(25,25)
#        self.linear2 = nn.Linear(25,25)

    def forward(self, x): 
        N, C, T, V = x.size()
        x1 = x[:,:C//2,:,:]
        x2 = x[:,C//2:C,:,:]
        Q_o = Q_Spa_Trans = self.avg_pool(x1.permute(0,3,1,2).contiguous())
        K_o = K_Spa_Trans = self.avg_pool(x2.permute(0,3,1,2).contiguous())
        Q_Spa_Trans = self.relu(self.linear(Q_Spa_Trans.squeeze(-1).squeeze(-1)))
        K_Spa_Trans = self.relu(self.linear(K_Spa_Trans.squeeze(-1).squeeze(-1)))
        Spa_atten = self.soft(torch.einsum('nv,nw->nvw', (Q_Spa_Trans, K_Spa_Trans))).unsqueeze(1).repeat(1,self.out_channels,1,1)  
        return Spa_atten, Q_o, K_o

class Spatial_MixFormer(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups=8, coff_embedding=4, num_subset=3,t_stride=1,t_padding=0,t_dilation=1,bias=True,first=False,residual=True):
        super(Spatial_MixFormer, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.groups=groups
        self.out_channels=out_channels
        self.in_channels=in_channels
        self.num_subset = 3
        self.alpha = nn.Parameter(torch.ones(1))
        self.A_GEME = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32),[3,1,25,25]), dtype=torch.float32, requires_grad=True).repeat(1,groups,1,1), requires_grad=True)
        self.A_SE = Variable(torch.from_numpy(np.reshape(A.astype(np.float32),[3,1,25,25]).repeat(groups,axis=1)), requires_grad=False) 
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(25,25)
        
        rr = 2
        self.fc1c = nn.Linear(out_channels, out_channels // rr)
        self.fc2c = nn.Linear(out_channels // rr, out_channels)
        self.fc3c = nn.Linear(out_channels, out_channels)
        
        
        self.Spa_Att = Spa_Atten(out_channels//4)
        self.AvpChaRef = nn.AdaptiveAvgPool2d(1) 
        self.ChaRef_conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False) 
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * num_subset,
            kernel_size=(1, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        self.fc = nn.Linear(50, 25)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / 25))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-1)
        self.relu = nn.ReLU()         
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x0):
        
        N, C, T, V = x0.size()
        A = self.A_SE + self.A_GEME
        norm_learn_A = A.repeat(1,self.out_channels//self.groups,1,1)      
        A_final=torch.zeros([N,self.num_subset,self.out_channels,25,25],dtype=torch.float).detach()        
        m = x0         
        m = self.conv(m)
        n, kc, t, v = m.size()
        m = m.view(n, self.num_subset, kc// self.num_subset, t, v)
        for i in range(self.num_subset):  
            m1,Q1,K1 = self.Spa_Att(m[:,i,:(kc// self.num_subset)//4,:,:])             
            m2,Q2,K2 = self.Spa_Att(m[:,i,(kc// self.num_subset)//4:((kc// self.num_subset)//4)*2,:,:])  
            m3,Q3,K3 = self.Spa_Att(m[:,i,((kc// self.num_subset)//4)*2:((kc// self.num_subset)//4)*3,:,:])             
            m4,Q4,K4 = self.Spa_Att(m[:,i,((kc// self.num_subset)//4)*3:((kc// self.num_subset)//4)*4,:,:])
            m1_2 = self.soft(torch.einsum('nv,nw->nvw', (self.relu(self.linear(K1.squeeze(-1).squeeze(-1))), self.relu(self.linear(Q2.squeeze(-1).squeeze(-1)))))).unsqueeze(1).repeat(1,self.out_channels//4,1,1)
            m2_3 = self.soft(torch.einsum('nv,nw->nvw', (self.relu(self.linear(K2.squeeze(-1).squeeze(-1))), self.relu(self.linear(Q3.squeeze(-1).squeeze(-1)))))).unsqueeze(1).repeat(1,self.out_channels//4,1,1)
            m3_4 = self.soft(torch.einsum('nv,nw->nvw', (self.relu(self.linear(K3.squeeze(-1).squeeze(-1))), self.relu(self.linear(Q4.squeeze(-1).squeeze(-1)))))).unsqueeze(1).repeat(1,self.out_channels//4,1,1)

            m1 = m1/2 + m1_2/4
            m2 = m2/2 + m1_2/4 + m2_3/4 
            m3 = m3/2 + m2_3/4 + m3_4/4 
            m4 = m4/2 + m3_4/4
            atten = torch.cat([m1,m2,m3,m4],dim=1)
            A_final[:,i,:,:,:] = atten * 0.5 + norm_learn_A[i]
        m = torch.einsum('nkctv,nkcvw->nctw', (m, A_final))   
        
        # Channel Reforming       
        CR_in = self.AvpChaRef(m)
        CR_in = self.ChaRef_conv(CR_in.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        CR_out = m + m * self.sigmoid(CR_in).expand_as(m)               
              
        out = self.bn(CR_out)        
        out += self.down(x0) 
        out = self.relu(out)
        return out

class unit_skip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_skip, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class Ske_MixF(nn.Module):
    def __init__(self, in_channels, out_channels, A, Frames, stride=1, residual=True):
        super(Ske_MixF, self).__init__()
        self.spa_mixf = Spatial_MixFormer(in_channels, out_channels, A)
        self.tem_mixf = Temporal_MixFormer(out_channels, out_channels, Frames, kernel_size=5, stride=stride, dilations=[1,2],residual=False)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_skip(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tem_mixf(self.spa_mixf(x)) + self.residual(x)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph()
        A = self.graph.A
        self.A_vector = self.get_A(graph, 1)
        self.num_point = num_point        
        self.data_bn = nn.BatchNorm1d(num_person * 80 * num_point)        
        self.to_joint_embedding = nn.Linear(in_channels, 80)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, 80))
        
        self.l1 = Ske_MixF(80, 80, A, 64, residual=False)
        self.l2 = Ske_MixF(80, 80, A, 64)
        self.l3 = Ske_MixF(80, 80, A, 64)
        self.l4 = Ske_MixF(80, 80, A, 64)
        self.l5 = Ske_MixF(80, 160, A, 32, stride=2)
        self.l6 = Ske_MixF(160, 160, A, 32)
        self.l7 = Ske_MixF(160, 160, A, 32)
        self.l8 = Ske_MixF(160, 320, A, 16, stride=2)
        self.l9 = Ske_MixF(320, 320, A, 16)
        self.l10= Ske_MixF(320, 320, A, 16)

        self.fc = nn.Linear(320, num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        
        # Retrospect Model
        self.first_tram = nn.Sequential(
                nn.AvgPool2d((4,1)),
                nn.Conv2d(80, 320, 1),
                nn.BatchNorm2d(320),
                nn.ReLU()
            )
        self.second_tram = nn.Sequential(
                nn.AvgPool2d((2,1)),
                nn.Conv2d(160, 320, 1),
                nn.BatchNorm2d(320),
                nn.ReLU()
            )
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        self.num_class=num_class
        
    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))        

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()

        p = self.A_vector
        p = torch.tensor(p,dtype=torch.float)
        x = p.expand(N*M*T, -1, -1) @ x
        
        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, :self.num_point]
        
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x2=x
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x3=x
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
                
        x2 = self.first_tram(x2)
        x3 = self.second_tram(x3)
        x =x + x2 + x3
        
        x = x.reshape(N, M, 320, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)


if __name__ == "__main__":
    # For debugging purposes
    import sys
    sys.path.append('..')

    model = Model(
        num_class=120,
        num_point=25,
        num_person=2,
        graph='graph.ntu_rgb_d.Graph'
    )

    N, C, T, V, M = 1, 3, 64, 25, 2
    x = torch.randn(N,C,T,V,M)
    model.forward(x)

    flops, params = thop.profile(model, inputs = (x,), verbose=False)
    #print(out.shape)
    print("flops",flops/1e9)
    print("params",params)

