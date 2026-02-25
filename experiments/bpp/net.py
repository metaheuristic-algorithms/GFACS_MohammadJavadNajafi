import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn
import torch_geometric


# GNN for edge embeddings
class GatedGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.u = nn.Linear(input_dim, output_dim)
        self.v = nn.Linear(input_dim, output_dim)
        self.A = nn.Linear(input_dim, output_dim)
        self.B = nn.Linear(input_dim, output_dim)
        self.C = nn.Linear(input_dim, output_dim)

        self.bn_node = gnn.BatchNorm(output_dim)
        self.bn_edge = gnn.BatchNorm(output_dim)

    def forward(self, x, edge_index, edge_attr):
        # x: (n, dim)
        # edge_index: (2, m)
        # edge_attr: (m, dim)

        row, col = edge_index

        # Edge update
        # e_ij = e_ij + ReLU(BN(A e_ij + B h_i + C h_j))
        edge_in = self.A(edge_attr) + self.B(x[row]) + self.C(x[col])
        edge_tmp = F.relu(self.bn_edge(edge_in))
        edge_attr = edge_attr + edge_tmp

        # Node update
        # h_i = h_i + ReLU(BN(U h_i + sum_j(sigmoid(e_ij) * V h_j)))
        sigma_edge = torch.sigmoid(edge_attr)
        node_neighbor = self.v(x)  # (n, dim)
        # We need to aggregate: sum_{j in N(i)} sigma_ij * V h_j
        # We can use scatter add or simpler pyg tools
        # V h_j is indexed by col
        msg = sigma_edge * node_neighbor[col]
        # Aggregate messages to row (target node)
        agg_msg = torch_geometric.utils.scatter(
            msg, row, dim=0, dim_size=x.size(0), reduce='sum')

        node_in = self.u(x) + agg_msg
        node_tmp = F.relu(self.bn_node(node_in))
        x = x + node_tmp

        return x, edge_attr


class GatedGCNEmbNet(nn.Module):
    def __init__(self, depth=12, feats=1, units=32, act_fn='silu', agg_fn='mean'):
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.units = units
        # Initial embeddings
        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.e_lin0 = nn.Linear(1, self.units)

        self.layers = nn.ModuleList(
            [GatedGCNLayer(self.units, self.units) for i in range(self.depth)])
        self.act_fn = getattr(F, act_fn)

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, x, edge_index, edge_attr):
        x = self.v_lin0(x)
        x = self.act_fn(x)
        edge_attr = self.e_lin0(edge_attr)
        edge_attr = self.act_fn(edge_attr)

        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)

        return edge_attr


# Original EmbNet
class OriginalEmbNet(nn.Module):
    def __init__(self, depth=12, feats=1, units=32, act_fn='silu', agg_fn='mean'):
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')
        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.v_lins1 = nn.ModuleList(
            [nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins2 = nn.ModuleList(
            [nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3 = nn.ModuleList(
            [nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4 = nn.ModuleList(
            [nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_bns = nn.ModuleList(
            [gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin0 = nn.Linear(1, self.units)
        self.e_lins0 = nn.ModuleList(
            [nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.e_bns = nn.ModuleList(
            [gnn.BatchNorm(self.units) for i in range(self.depth)])

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, x, edge_index, edge_attr):
        x = x
        w = edge_attr
        x = self.v_lin0(x)
        x = self.act_fn(x)
        w = self.e_lin0(w)
        w = self.act_fn(w)
        for i in range(self.depth):
            x0 = x
            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)
            w0 = w
            w1 = self.e_lins0[i](w0)
            w2 = torch.sigmoid(w0)
            x = x0 + \
                self.act_fn(self.v_bns[i](
                    x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
            w = w0 + \
                self.act_fn(self.e_bns[i](
                    w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return w


# general class for MLP
class MLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device

    def __init__(self, units_list, act_fn):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad=False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        self.lins = nn.ModuleList(
            [nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])

    def forward(self, x):
        for i in range(self.depth):
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
            else:
                x = torch.sigmoid(x)  # last layer
        return x


# MLP for predicting parameterization theta
class ParNet(MLP):
    def __init__(self, depth=3, units=32, preds=1, act_fn='silu'):
        self.units = units
        self.preds = preds
        super().__init__([self.units] * depth + [self.preds], act_fn)

    def forward(self, x):
        return super().forward(x).squeeze(dim=-1)


class Net(nn.Module):
    def __init__(self, gfn=False, Z_out_dim=1, model_type='gcn'):
        super().__init__()
        if model_type == 'gcn':
            self.emb_net = GatedGCNEmbNet()
        elif model_type == 'embnet':
            self.emb_net = OriginalEmbNet()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.par_net_heu = ParNet()

        self.gfn = gfn
        self.Z_net = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, Z_out_dim),
        ) if gfn else None

    def forward(self, pyg, return_logZ=False):
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr
        emb = self.emb_net(x, edge_index, edge_attr)
        heu = self.par_net_heu(emb)

        if return_logZ:
            assert self.gfn and self.Z_net is not None
            logZ = self.Z_net(emb).mean(0)
            return heu, logZ

        return heu

    def freeze_gnn(self):
        for param in self.emb_net.parameters():
            param.requires_grad = False

    @staticmethod
    def reshape(pyg, vector):
        '''Turn phe/heu vector into matrix with zero padding'''
        n_nodes = pyg.x.shape[0]
        device = pyg.x.device
        matrix = torch.zeros(size=(n_nodes, n_nodes), device=device)
        matrix[pyg.edge_index[0], pyg.edge_index[1]] = vector
        return matrix
