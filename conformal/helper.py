import torch
from torch.nn.parameter import Parameter
from torch import nn,optim
import torch.nn.functional as F
import math
import copy
import torch_geometric
from torch_geometric.nn.conv import HypergraphConv
from torch_geometric.nn import LayerNorm
from torch_geometric.utils import scatter
import numpy as np
import scipy
from collections import defaultdict


def _as_int(value):
    """Return Python ints for PyG attributes that may be tensors or scalars."""
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    return int(value)


def permute_edges(data, aug_ratio, permute_self_edge, device):
    """Drop a fraction of node-to-hyperedge incidences for contrastive augmentation.

    PyG hypergraph data in this project stores real hyperedges first and appends
    self-loop hyperedges after ``data.num_hyperedges``.  By default, only the
    real incidences are sampled away so each node's self-edge remains available.
    """
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    hyperedge_num = _as_int(data.num_hyperedges)
    edge_index = data.edge_index.cpu().numpy()

    if permute_self_edge:
        edge2remove_index = np.arange(edge_num)
        edge2keep_index = np.array([], dtype=int)
    else:
        edge2remove_index = np.where(edge_index[1] < hyperedge_num)[0]
        edge2keep_index = np.where(edge_index[1] >= hyperedge_num)[0]

    keep_num = int(len(edge2remove_index) * (1 - aug_ratio))
    keep_num = min(max(keep_num, 0), len(edge2remove_index))
    edge_keep_index = np.random.choice(edge2remove_index, keep_num, replace=False)
    edge_after_remove1 = edge_index[:, edge_keep_index]
    edge_after_remove2 = edge_index[:, edge2keep_index]

    edge_index = np.concatenate((edge_after_remove1, edge_after_remove2),axis=1)
    edge_index = torch.from_numpy(edge_index).long().to(device)
    return edge_index

def permute_hyperedges(data, aug_ratio, device):
    """Drop complete hyperedges for contrastive augmentation.

    All incidences belonging to sampled hyperedge ids are removed together,
    preserving the incidence structure of hyperedges that remain.
    """

    _, edge_num = data.edge_index.size()
    hyperedge_num = _as_int(data.num_hyperedges)

    permute_num = int(hyperedge_num * aug_ratio)
    permute_num = min(max(permute_num, 0), hyperedge_num)
    edge_index = data.edge_index.cpu().numpy()
    edge_remove_index = np.random.choice(hyperedge_num, permute_num, replace=False)
    edge_remove_index_dict = {ind: i for i, ind in enumerate(edge_remove_index)}

    edge_remove_index_all = [i for i, he in enumerate(edge_index[1]) if he in edge_remove_index_dict]
    # print(len(edge_remove_index_all), edge_num, len(edge_remove_index), aug_ratio, hyperedge_num)
    edge_keep_index = list(set(list(range(edge_num)))-set(edge_remove_index_all))
    edge_after_remove = edge_index[:, edge_keep_index]
    edge_index = torch.tensor(edge_after_remove).long().to(device)
    return edge_index

class HGNN(nn.Module):
    """Two-layer hypergraph neural network used as the base classifier."""

    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HypergraphConv(in_channels=in_ch, out_channels=n_hid)
        self.hgc2 = HypergraphConv(in_channels=n_hid, out_channels=n_class)
        self.reset_parameters()
    def reset_parameters(self):
        self.hgc1.reset_parameters()
        self.hgc2.reset_parameters()

    def forward(self, data):
        x = data.x
        G = data.edge_index
        #print("x.shape:", x.shape)  # Should be [num_nodes, num_features]
        #print("edge_index shape:", G.shape)  # Should be [2, num_edges]
        #print("max node index in edge_index[0]:", G[0].max().item())
        #print("max node index in edge_index[1]:", G[1].max().item())
        #import pdb; pdb.set_trace()

        x=self.hgc1(x, G)
        # Keep dropout tied to module mode; otherwise evaluation remains random.
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc2(x, G)
        return x
class HNN_Multi_Layer(torch.nn.Module):
    """Stacked HypergraphConv block used by conformal correction heads."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers = 2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            self.convs.append(HypergraphConv(in_channels=in_channels, out_channels=out_channels))
        else:
            self.convs.append(HypergraphConv(in_channels=in_channels, out_channels=hidden_channels))

            for _ in range(num_layers-2):

                self.convs.append(HypergraphConv(in_channels=hidden_channels, out_channels=hidden_channels))

            self.convs.append(HypergraphConv(in_channels=hidden_channels, out_channels=out_channels))
        self.normalization = LayerNorm(hidden_channels)
    def forward(self, x, edge_index):
        for idx, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if idx == len(self.convs) - 1:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index).relu()
                x = self.normalization(x)
        return x


class SimpleMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.FC_hidden = nn.Linear(input_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        h     = self.ReLU(self.FC_hidden(x))
        h     = self.ReLU(self.FC_hidden2(h))
        x_hat = self.FC_output(h)
        return x_hat
class ConfHNN(torch.nn.Module):
    """Conformal HNN head trained with incidence-level edge augmentations."""

    def __init__(self, model, dataset, args, num_conf_layers, output_dim):
        super().__init__()
        self.model = model
        self.data=dataset
        self.args = args
        #num_classes = max(dataset.y).item() + 1
        #print(base_model)
        self.confhnn = HNN_Multi_Layer(output_dim, args.confnn_hidden_dim, output_dim, num_conf_layers, dropout=getattr(args, 'confgnn_dropout', 0.5))
    def forward(self, x, edge_index,argu=None):

        with torch.no_grad():
            scores = self.model(self.data)

        out = F.softmax(scores, dim = 1)
        # During contrastive training, build two stochastic graph views.
        if argu is not None:
            adjust_scores = self.confhnn(out, edge_index)
            return adjust_scores, scores
        aug_ratio = getattr(self.args, 'aug_ratio', 0.3)
        edge_index1=permute_edges(self.data, aug_ratio, False, self.args.device)
        edge_index2=permute_edges(self.data, aug_ratio, False, self.args.device)
        adjust_scores1 = self.confhnn(out, edge_index1)
        adjust_scores2 = self.confhnn(out, edge_index2)
        return adjust_scores1,adjust_scores2, scores

class ConfHNN1(torch.nn.Module):
    """Conformal HNN head trained with whole-hyperedge augmentations."""

    def __init__(self, model, dataset, args, num_conf_layers, output_dim):
        super().__init__()
        self.model = model
        self.data=dataset
        self.args = args
        #num_classes = max(dataset.y).item() + 1
        #print(base_model)
        self.confhnn = HNN_Multi_Layer(output_dim, args.confnn_hidden_dim, output_dim, num_conf_layers, dropout=getattr(args, 'confgnn_dropout', 0.5))
    def forward(self, x, edge_index,argu=None):

        with torch.no_grad():
            scores = self.model(self.data)

        out = F.softmax(scores, dim = 1)
        # During contrastive training, build two stochastic graph views.
        if argu is not None:
            adjust_scores = self.confhnn(out, edge_index)
            return adjust_scores, scores
        aug_ratio = getattr(self.args, 'aug_ratio', 0.3)
        edge_index1=permute_hyperedges(self.data, aug_ratio, self.args.device)
        edge_index2=permute_hyperedges(self.data, aug_ratio, self.args.device)
        adjust_scores1 = self.confhnn(out, edge_index1)
        adjust_scores2 = self.confhnn(out, edge_index2)
        return adjust_scores1,adjust_scores2, scores

class ConfMLP(torch.nn.Module):
    """MLP conformal head; augmentation calls are kept for API parity."""

    def __init__(self, model, dataset, args, output_dim):
        super().__init__()
        self.model = model
        self.confmlp = SimpleMLP(output_dim, getattr(args, 'confnn_hidden_dim', 64), output_dim)
        self.data=dataset
        self.args = args
    def forward(self, x, edge_index,argu=None):

        with torch.no_grad():
            scores = self.model(self.data)

        out = F.softmax(scores, dim = 1)
        # The MLP ignores edges, but callers expect the same tuple as ConfHNN.
        aug_ratio = getattr(self.args, 'aug_ratio', 0.3)
        edge_index1=permute_edges(self.data, aug_ratio, False, self.args.device)
        edge_index2=permute_edges(self.data, aug_ratio, False, self.args.device)
        adjust_scores1 = self.confmlp(out)
        adjust_scores2 = self.confmlp(out)
        return adjust_scores1,adjust_scores2, scores

class edge_task(torch.nn.Module):
    """Learn a differentiable top-k hyperedge mask from incidence features."""

    def __init__(self, data, embed_dim):
        super().__init__()
        x = data.x
        G = data.edge_index
        num_edges = int(G[1].max()) + 1
        num_nodes = x.size(0)
        # H: [num_nodes, num_edges], incidence matrix (sparse)
        indices = torch.stack([G[0], G[1]], dim=0)
        values = torch.ones(G.size(1), device=x.device)
        self.H = torch.sparse_coo_tensor(indices, values, (num_nodes, num_edges), device=x.device)
        in_features = x.shape[1]  # Number of input features per node
        #self.attn = nn.MultiheadAttention(in_features, num_heads=4)
        #self.feed=nn.Linear(in_features,embed_dim)
        # make x sparse and laplacian is h*x
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2)
        )
        self.laplacian = self.H.T @ x
    def attention(self):
        # Each column is a hyperedge
        # Compute a feature for each hyperedge (e.g., deviation from mean)
        mean = self.laplacian.mean(dim=0, keepdim=True)  # (1, n)
        deviation = self.laplacian - mean  # (m, n)

        # Aggregate deviation as hyperedge feature (e.g., L2 norm)
        #hyperedge_feat = deviation.norm(dim=0).unsqueeze(1)  # (n, 1)
        # Compute attention scores via MLP
        self.attn_weights = self.attn_mlp(deviation)  # (n, 1)
        #import pdb; pdb.set_trace()
        self.attn_weights = self.attn_weights.squeeze(1)  # (n,)
        #return attn_scores
    """ def attention(self):  # laplacian: (m, n) each column is a hyperedge
        # Let's use each column as a feature for each hyperedge
        #hyperedge_feats = self.feed(self.laplacian.T)  # (n, m) treat each column as a feature vector
        #import pdb; pdb.set_trace()
        # Project to key, query, value
        _, self.attn_weights = self.attn(self.laplacian, self.laplacian, self.laplacian)  # (n, m), (n, m), (n, m)

        # Compute attention scores (dot product: query vs key)
        # For self-attention, each hyperedge attends to all others
        #scores = torch.matmul(queries, keys.T) / (keys.shape[-1] ** 0.5)  # (n, n)

        # Softmax over keys for each query
        #self.attn_weights = F.softmax(scores, dim=-1)  # (n, n)

        # Output: weighted sum of values
        #self.attended = torch.matmul(self.attn_weights, values)  # (n, embed_dim)

 """
    def forward(self, k, tau=0.5):
        if not hasattr(self, "attn_weights"):
            self.attention()
        # attn_scores: (n,) or (n, n), we want a mask for top-k
        # For simplicity, let's aggregate to (n,) by mean/diag or use a column
        #if self.attn_weights.dim() == 2:
        attn_scores = self.attn_weights.mean(dim=1)  # (n,)
        y = F.gumbel_softmax(attn_scores, tau=tau, hard=False)
        topk_mask = torch.zeros_like(y)
        #import pdb; pdb.set_trace()
        _, idx = torch.topk(y, k)
        topk_mask[idx] = y[idx]

        return topk_mask,self.H.T
    """ def forward(self, k, tau=0.5):
        # attn_scores: (n,) or (n, n), we want a mask for top-k
        # For simplicity, let's aggregate to (n,) by mean/diag or use a column
        if self.attn_weights.dim() == 2:
            attn_scores = self.attn_weights.mean(dim=0)  # (n,)
        else:
            attn_scores = self.attn_weights  # (n,)
        y = F.gumbel_softmax(attn_scores, tau=tau, hard=False)
        topk_mask = torch.zeros_like(y)
        _, idx = torch.topk(y, k)
        topk_mask[idx] = y[idx]
        return topk_mask, self.H.T """

class HyperedgeDegreePredictor(nn.Module):
    """Small regressor for predicting scalar hyperedge degree statistics."""

    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, hyperedge_features):
        pred = self.mlp(hyperedge_features)  # (k, 1)
        return pred.squeeze(1)
def fit_calibration(temp_model, eval, data, train_mask, test_mask, patience = 100):
    """
    Train calibrator
    """
    vlss_mn = float('Inf')
    with torch.no_grad():
        logits = temp_model.model(data)
        labels = data.y
        edge_index = data.edge_index
        model_dict = temp_model.state_dict()
        parameters = {k: v for k,v in model_dict.items() if k.split(".")[0] != "model"}
    for epoch in range(2000):
        temp_model.optimizer.zero_grad()
        temp_model.train()
        # Post-hoc calibration set the classifier to the evaluation mode
        temp_model.model.eval()
        assert not temp_model.model.training
        calibrated = eval(logits)
        loss = F.cross_entropy(calibrated[train_mask], labels[train_mask])
        # dist_reg = intra_distance_loss(calibrated[train_mask], labels[train_mask])
        # margin_reg = 0.
        # loss = loss + margin_reg * dist_reg
        loss.backward()
        temp_model.optimizer.step()

        with torch.no_grad():
            temp_model.eval()
            calibrated = eval(logits)
            val_loss = F.cross_entropy(calibrated[test_mask], labels[test_mask])
            # dist_reg = intra_distance_loss(calibrated[train_mask], labels[train_mask])
            # val_loss = val_loss + margin_reg * dist_reg
            if val_loss <= vlss_mn:
                # Re-read the state dict so early stopping stores current values.
                state_dict_early_model = copy.deepcopy({
                    k: v for k, v in temp_model.state_dict().items()
                    if k.split(".")[0] != "model"
                })
                vlss_mn = np.min((val_loss.cpu().numpy(), vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step >= patience:
                    break
    model_dict.update(state_dict_early_model)
    temp_model.load_state_dict(model_dict)

class TS(nn.Module):
    """Post-hoc temperature scaling calibrator."""

    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))
        self.device = device
    def forward(self, x, edge_index):
        # Create a mock data object for the HGNN model
        class MockData:
            def __init__(self, x, edge_index):
                self.x = x
                self.edge_index = edge_index

        data = MockData(x, edge_index)
        logits = self.model(data)
        temperature = self.temperature_scale(logits)
        return logits / temperature

    def temperature_scale(self, logits):
        """
        Expand temperature to match the size of logits
        """
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return temperature

    def fit(self, data, train_mask, test_mask, wdecay):
        self.to(self.device)
        def eval(logits):
            temperature = self.temperature_scale(logits)
            calibrated = logits / temperature
            return calibrated

        self.train_param = [self.temperature]
        self.optimizer = optim.Adam(self.train_param, lr=0.01, weight_decay=wdecay)
        fit_calibration(self, eval, data, train_mask, test_mask)
        return self

class VS(nn.Module):
    """Vector scaling calibrator with class-wise scales and biases."""

    def __init__(self, model, num_classes, device):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(num_classes))
        self.bias = nn.Parameter(torch.ones(num_classes))
        self.device = device
    def forward(self, x, edge_index):
        # Create a mock data object for the HGNN model
        class MockData:
            def __init__(self, x, edge_index):
                self.x = x
                self.edge_index = edge_index

        data = MockData(x, edge_index)
        logits = self.model(data)
        temperature = self.vector_scale(logits)
        return logits * temperature + self.bias

    def vector_scale(self, logits):
        """
        Expand temperature to match the size of logits
        """
        temperature = self.temperature.unsqueeze(0).expand(logits.size(0), logits.size(1))
        return temperature

    def fit(self, data, train_mask, test_mask, wdecay):
        self.to(self.device)
        def eval(logits):
            temperature = self.vector_scale(logits)
            calibrated = logits * temperature + self.bias
            return calibrated

        self.train_param = [self.temperature, self.bias]
        self.optimizer = optim.Adam(self.train_param, lr=0.01, weight_decay=wdecay)
        fit_calibration(self, eval, data, train_mask, test_mask)
        return self

class ETS(nn.Module):
    """Ensemble temperature scaling calibrator."""

    def __init__(self, model, num_classes, device):
        super().__init__()
        self.model = model
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.zeros(1))
        self.w3 = nn.Parameter(torch.zeros(1))
        self.num_classes = num_classes
        self.temp_model = TS(model, device)
        self.device = device
    def forward(self, x, edge_index):
        # Create a mock data object for the HGNN model
        class MockData:
            def __init__(self, x, edge_index):
                self.x = x
                self.edge_index = edge_index

        data = MockData(x, edge_index)
        logits = self.model(data)
        temp = self.temp_model.temperature_scale(logits)
        p = self.w1 * F.softmax(logits / temp, dim=1) + self.w2 * F.softmax(logits, dim=1) + self.w3 * 1/self.num_classes
        return torch.log(p)

    def fit(self, data, train_mask, test_mask, wdecay):
        self.to(self.device)
        self.temp_model.fit(data, train_mask, test_mask, wdecay)
        torch.cuda.empty_cache()
        logits = self.model(data)[train_mask]
        label = data.y[train_mask]
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.unsqueeze(-1), 1)
        temp = self.temp_model.temperature.cpu().detach().numpy()
        w = self.ensemble_scaling(logits.cpu().detach().numpy(), one_hot.cpu().detach().numpy(), temp)
        with torch.no_grad():
            self.w1.fill_(float(w[0]))
            self.w2.fill_(float(w[1]))
            self.w3.fill_(float(w[2]))
        return self

    def ensemble_scaling(self, logit, label, t):
        """
        Official ETS implementation from Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning
        Code taken from (https://github.com/zhang64-llnl/Mix-n-Match-Calibration)
        Use the scipy optimization because PyTorch does not have constrained optimization.
        """
        p1 = np.exp(logit)/np.sum(np.exp(logit),1)[:,None]
        logit = logit/t
        p0 = np.exp(logit)/np.sum(np.exp(logit),1)[:,None]
        p2 = np.ones_like(p0)/self.num_classes


        bnds_w = ((0.0, 1.0),(0.0, 1.0),(0.0, 1.0),)
        def my_constraint_fun(x): return np.sum(x)-1
        constraints = { "type":"eq", "fun":my_constraint_fun,}
        w = scipy.optimize.minimize(ETS.ll_w, (1.0, 0.0, 0.0), args = (p0,p1,p2,label), method='SLSQP', constraints = constraints, bounds=bnds_w, tol=1e-12, options={'disp': False})
        w = w.x
        return w

    @staticmethod
    def ll_w(w, *args):
    ## find optimal weight coefficients with Cros-Entropy loss function
        p0, p1, p2, label = args
        p = (w[0]*p0+w[1]*p1+w[2]*p2)
        N = p.shape[0]
        ce = -np.sum(label*np.log(p))/N
        return ce
