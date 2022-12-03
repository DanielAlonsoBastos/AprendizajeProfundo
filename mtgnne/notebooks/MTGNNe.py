from __future__ import division

import numbers
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    r"""
    Implementation of a Graph Attention Layer
    Based on: https://nn.labml.ai/graphs/gat/index.html

    Args:
        in_features (int): Number of input features per node
        out_features (int): Number of output features per node
        n_heads (int): Number of attention heads
        concat (bool): Whether the multi-head results should be concatenated or averaged
        dropout (float): Dropout probability
        negative_slope (float): Negative slope for leaky relu activation
        bias (bool): Include bias term
    """
    def __init__(self, in_features, out_features, n_heads, dropout=0.3, concat=True, negative_slope=0.2, bias=True):
        super(GraphAttentionLayer, self).__init__()
        
        self.concat = concat
        self.n_heads = n_heads
        
        # calcular el numero de dimensiones por cabeza
        if concat:
            assert out_features % n_heads == 0
            # numero de unidades ocultas
            self.n_hidden = out_features // n_heads # division que regresa un entero
        else:
            # promedio de las multiples cabezas
            self.n_hidden = out_features
        
        # capa lineal para la transformacion inicial -> transformar el embedding de los nodos antes de self-attention
        #in_features = B*F*S
        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=bias)
        # capa lineal para calcular el score de atencion e_ij
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=bias)
        self.activation = nn.LeakyReLU(negative_slope)
        # softmax para calcular la atencion
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    
    def forward(self, X, A):
        # X -> [B, F, N, S] , B->batch, F->features, N->num_nodes, S->seq(window)
        batch_size, features, n_nodes, seq = X.shape
        # [B, F, N, S] -> [B, S, N, F] -> dejar F como ultimo elemento
        X = torch.permute(X, [0, 3, 2, 1])
        # aplicar la primera transformacion a cada cabeza g = Wh
        # torch.view -> regresa un tensor con la misma data pero de diferente tamanio
        # esto es para dividirlo para cada cabeza
        g = self.linear(X).view(batch_size, seq, n_nodes, self.n_heads, self.n_hidden) # [B, S, N, H, F]
        # calculo de e_ij para cada par: e = alpha(Wh_i, W_hj) = alpha(g_i,g_j)
        ## calcular [g_i||g_j] para cada par
        g_repeat = g.repeat(1, 1, n_nodes, 1, 1)
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=2)
        # concatenar
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        # redimensionar [B, S, N, N, H, 2*F]
        g_concat = g_concat.view(batch_size, seq, n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)
        # calcular con la funcion de activacion e = LeakyReLU(aT[g_i||g_j])
        # [B, S, N, N, H, 2*F] -> [B, S, N, N, H, 1]
        e = self.activation(self.attn(g_concat))
        # quitar la ultima dimension de tamanio 1
        # [B, S, N, N, H, 1] -> [B, S, N, N, H]
        e = e.squeeze(-1)
        # la matriz de adyacencia A debe tener la siguiente forma: [n_nodes, n_nodes, 1]
        A_matrix = torch.unsqueeze(A, dim=-1)
        assert A_matrix.shape[0] == 1 or A_matrix.shape[0] == n_nodes
        assert A_matrix.shape[1] == 1 or A_matrix.shape[1] == n_nodes
        assert A_matrix.shape[2] == 1 or A_matrix.shape[2] == self.n_heads
        # mascara en A (no existe alguna arista entre los nodos i,j)
        # [B, S, N, N, H]
        e = e.masked_fill(A_matrix  == 0, float('-inf'))
        # normalizar los coeficientes de atencion: alphas = softmax(e)
        alphas = self.softmax(e) # [B, S, N, N, H]
        # remplazar los nan por cero
        alphas = torch.nan_to_num(alphas, nan=0.0)
        alphas = self.dropout(alphas) # [B, S, N, N, H]
        # resultado para cada cabeza
        # alphas -> [B, S, N, N, H]
        # g -> [B, S, N, H, F]
        attn_res = torch.einsum('mnojh,klphf->mnohf', alphas, g) # [B, S, N, H, F]
        
        # manipulacion de las cabezas
        if self.concat:
            # concatenar
            # [B, S, N, H, F] -> [B, S, N, F]
            attn_res = attn_res.reshape(batch_size, seq, n_nodes, self.n_heads * self.n_hidden)
            
        else:
            # promediar
            # [B, S, N, H, F] -> [B, S, N, F]
            attn_res = attn_res.mean(dim=3) # promedio sobre las cabezaas
        
        # [B, S, N, F] -> [B, F, N, S] (para ser utilizado en la capa de convolucion)
        return torch.permute(attn_res, [0, 3, 2, 1])


class Linear(nn.Module):
    r"""An implementation of the linear layer, conducting 2D convolution.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        bias (bool, optional): Whether to have bias. Default: True.
    """

    def __init__(self, c_in: int, c_out: int, bias: bool = True):
        super(Linear, self).__init__()
        self._mlp = torch.nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the linear layer.

        Arg types:
            * **X** (Pytorch Float Tensor) - Input tensor, with shape (batch_size, c_in, num_nodes, seq_len).

        Return types:
            * **X** (PyTorch Float Tensor) - Output tensor, with shape (batch_size, c_out, num_nodes, seq_len).
        """
        return self._mlp(X)


class MixProp(nn.Module):
    r"""An implementation of the dynatic mix-hop propagation layer.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Modification to extend the use attention in the adjacency matrix (MTGAN) and/or between hops (MTGNNAH/MTGANAH)
    MTGNNAH/MTGANAH reference: https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/e43e95d726174574eeae32ad8c4670631fd0a58d/torch_geometric_temporal/nn/recurrent/attentiontemporalgcn.py#L7

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        gdep (int): Depth of graph convolution.
        dropout (float): Dropout rate.
        alpha (float): Ratio of retaining the root nodes's original states, a value between 0 and 1.
        arquitecture (str): Type of arquitecture to use ('MTGNN'->original, 'MTGAN'->attention in adj_matrix, 'MTGNNAH'->MTGNN with attention in hops,'MTGANAH'->MTGAN with attention in hops)
        n_heads (int): Number of attention heads, in case to use MTGAN
        concat (bool): Whether the multi-head results should be concatenated or averaged, in case to use MTGAN
    """

    def __init__(self, c_in: int, c_out: int, gdep: int, dropout: float, beta: float, arquitecture: str, n_heads: int, concat: bool):
        super(MixProp, self).__init__()
        self._mlp = Linear((gdep + 1) * c_in, c_out)
        self._gdep = gdep
        self._dropout = dropout
        self._beta = beta
        self._arq = arquitecture

        if self._arq == 'MTGAN' or self._arq == 'MTGANAH':
            self._gan = GraphAttentionLayer(c_in, c_out, n_heads, dropout, concat)
        
        if self._arq == 'MTGNNAH' or self._arq == 'MTGANAH':
            # tensor de atencion (tensor con parametros entrenables)
            self._attention = nn.Parameter(torch.empty(self._gdep, requires_grad=True))


        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor, A: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of mix-hop propagation.

        Arg types:
            * **X** (Pytorch Float Tensor) - Input feature Tensor, with shape (batch_size, c_in, num_nodes, seq_len).
            * **A** (PyTorch Float Tensor) - Adjacency matrix, with shape (num_nodes, num_nodes).

        Return types:
            * **H_0** (PyTorch Float Tensor) - Hidden representation for all nodes, with shape (batch_size, c_out, num_nodes, seq_len).
        """
        A = A + torch.eye(A.size(0)).to(X.device)
        d = A.sum(1)
        H = X
        H_0 = X
        A = A / d.view(-1, 1)

        if self._arq == 'MTGNNAH' or self._arq == 'MTGANAH':
            # normalizacion del tensor de atenciones
            probs = F.softmax(self._attention, dim=0)

        for k in range(self._gdep):
            if self._arq == 'MTGNN':
                Hk = torch.einsum("ncwl,vw->ncvl", (H, A))
            elif self._arq == 'MTGAN':
                Hk = self._gan(H,A)
            elif self._arq == 'MTGNNAH':
                Hk = probs[k] * torch.einsum("ncwl,vw->ncvl", (H, A))
            elif self._arq == 'MTGANAH':
                Hk = probs[k] * self._gan(H,A)
            else:
                raise RuntimeError(f'Invalid arquitecture: {self._arq}, available options: MTGNN, MTGAN, MTGNNAH, MTGANAH')
            H = self._beta * X + (1 - self._beta) * Hk
            H_0 = torch.cat((H_0, H), dim=1)
        H_0 = self._mlp(H_0)
        return H_0


class DilatedInception(nn.Module):
    r"""An implementation of the dilated inception layer.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        kernel_set (list of int): List of kernel sizes.
        dilated_factor (int, optional): Dilation factor.
    """

    def __init__(self, c_in: int, c_out: int, kernel_set: list, dilation_factor: int):
        super(DilatedInception, self).__init__()
        self._time_conv = nn.ModuleList()
        self._kernel_set = kernel_set
        c_out = int(c_out / len(self._kernel_set))
        for kern in self._kernel_set:
            self._time_conv.append(
                nn.Conv2d(c_in, c_out, (1, kern), dilation=(1, dilation_factor))
            )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X_in: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of dilated inception.

        Arg types:
            * **X_in** (Pytorch Float Tensor) - Input feature Tensor, with shape (batch_size, c_in, num_nodes, seq_len).

        Return types:
            * **X** (PyTorch Float Tensor) - Hidden representation for all nodes,
            with shape (batch_size, c_out, num_nodes, seq_len-6).
        """
        X = []
        for i in range(len(self._kernel_set)):
            X.append(self._time_conv[i](X_in))
        for i in range(len(self._kernel_set)):
            X[i] = X[i][..., -X[-1].size(3) :]
        X = torch.cat(X, dim=1)
        return X


class GraphConstructor(nn.Module):
    r"""An implementation of the graph learning layer to construct an adjacency matrix.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        nnodes (int): Number of nodes in the graph.
        k (int): Number of largest values to consider in constructing the neighbourhood of a node (pick the "nearest" k nodes).
        dim (int): Dimension of the node embedding.
        alpha (float, optional): Tanh alpha for generating adjacency matrix, alpha controls the saturation rate
        xd (int, optional): Static feature dimension, default None.
    """

    def __init__(
        self, nnodes: int, k: int, dim: int, alpha: float, xd: Optional[int] = None
    ):
        super(GraphConstructor, self).__init__()
        if xd is not None:
            self._static_feature_dim = xd
            self._linear1 = nn.Linear(xd, dim)
            self._linear2 = nn.Linear(xd, dim)
        else:
            self._embedding1 = nn.Embedding(nnodes, dim)
            self._embedding2 = nn.Embedding(nnodes, dim)
            self._linear1 = nn.Linear(dim, dim)
            self._linear2 = nn.Linear(dim, dim)

        self._k = k
        self._alpha = alpha

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self, idx: torch.LongTensor, FE: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        """
        Making a forward pass to construct an adjacency matrix from node embeddings.

        Arg types:
            * **idx** (Pytorch Long Tensor) - Input indices, a permutation of the number of nodes, default None (no permutation).
            * **FE** (Pytorch Float Tensor, optional) - Static feature, default None.
        Return types:
            * **A** (PyTorch Float Tensor) - Adjacency matrix constructed from node embeddings.
        """

        if FE is None:
            nodevec1 = self._embedding1(idx)
            nodevec2 = self._embedding2(idx)
        else:
            assert FE.shape[1] == self._static_feature_dim
            nodevec1 = FE[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self._alpha * self._linear1(nodevec1))
        nodevec2 = torch.tanh(self._alpha * self._linear2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(
            nodevec2, nodevec1.transpose(1, 0)
        )
        A = F.relu(torch.tanh(self._alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(A.device)
        mask.fill_(float("0"))
        s1, t1 = A.topk(self._k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        A = A * mask
        return A



class LayerNormalization(nn.Module):
    __constants__ = ["normalized_shape", "weight", "bias", "eps", "elementwise_affine"]
    r"""An implementation of the layer normalization layer.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks." 
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        normalized_shape (int): Input shape from an expected input of size.
        eps (float, optional): Value added to the denominator for numerical stability. Default: 1e-5.
        elementwise_affine (bool, optional): Whether to conduct elementwise affine transformation or not. Default: True.
    """

    def __init__(
        self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True
    ):
        super(LayerNormalization, self).__init__()
        self._normalized_shape = tuple(normalized_shape)
        self._eps = eps
        self._elementwise_affine = elementwise_affine
        if self._elementwise_affine:
            self._weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self._bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter("_weight", None)
            self.register_parameter("_bias", None)
        self._reset_parameters()

    def _reset_parameters(self):
        if self._elementwise_affine:
            init.ones_(self._weight)
            init.zeros_(self._bias)

    def forward(self, X: torch.FloatTensor, idx: torch.LongTensor) -> torch.FloatTensor:
        """
        Making a forward pass of layer normalization.

        Arg types:
            * **X** (Pytorch Float Tensor) - Input tensor,
                with shape (batch_size, feature_dim, num_nodes, seq_len).
            * **idx** (Pytorch Long Tensor) - Input indices.

        Return types:
            * **X** (PyTorch Float Tensor) - Output tensor,
                with shape (batch_size, feature_dim, num_nodes, seq_len).
        """
        if self._elementwise_affine:
            return F.layer_norm(
                X,
                tuple(X.shape[1:]),
                self._weight[:, idx, :],
                self._bias[:, idx, :],
                self._eps,
            )
        else:
            return F.layer_norm(
                X, tuple(X.shape[1:]), self._weight, self._bias, self._eps
            )


class MTGNNLayer(nn.Module):
    r"""An implementation of the MTGNN layer.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        dilation_exponential (int): Dilation exponential.
        rf_size_i (int): Size of receptive field.
        kernel_size (int): Size of kernel for convolution, to calculate receptive field size.
        j (int): Iteration index.
        residual_channels (int): Residual channels.
        conv_channels (int): Convolution channels.
        skip_channels (int): Skip channels.
        kernel_set (list of int): List of kernel sizes.
        new_dilation (int): Dilation.
        layer_norm_affline (bool): Whether to do elementwise affine in Layer Normalization.
        gcn_true (bool): Whether to add graph convolution layer.
        seq_length (int): Length of input sequence.
        receptive_field (int): Receptive field.
        dropout (float): Droupout rate.
        gcn_depth (int): Graph convolution depth.
        num_nodes (int): Number of nodes in the graph.
        propbeta (float): Prop beta, ratio of retaining the root nodes's original states in mix-hop propagation, a value between 0 and 1.
        arquitecture (str): Type of arquitecture to use ('MTGNN'->original, 'MTGAN'->attention in adj_matrix, 'MTGNNAH'->MTGNN with attention in hops,'MTGANAH'->MTGAN with attention in hops)
        n_heads (int): Number of attention heads, in case to use MTGAN
        concat (bool): Whether the multi-head results should be concatenated or averaged, in case to use MTGAN
    """

    def __init__(
        self,
        dilation_exponential: int,
        rf_size_i: int,
        kernel_size: int,
        j: int,
        residual_channels: int,
        conv_channels: int,
        skip_channels: int,
        kernel_set: list,
        new_dilation: int,
        layer_norm_affline: bool,
        gcn_true: bool,
        seq_length: int,
        receptive_field: int,
        dropout: float,
        gcn_depth: int,
        num_nodes: int,
        propbeta: float,
        arquitecture: str,
        n_heads: int,
        concat: bool,
    ):
        super(MTGNNLayer, self).__init__()
        self._dropout = dropout
        self._gcn_true = gcn_true

        if dilation_exponential > 1:
            rf_size_j = int(
                rf_size_i
                + (kernel_size - 1)
                * (dilation_exponential ** j - 1)
                / (dilation_exponential - 1)
            )
        else:
            rf_size_j = rf_size_i + j * (kernel_size - 1)

        self._filter_conv = DilatedInception(
            residual_channels,
            conv_channels,
            kernel_set=kernel_set,
            dilation_factor=new_dilation,
        )

        self._gate_conv = DilatedInception(
            residual_channels,
            conv_channels,
            kernel_set=kernel_set,
            dilation_factor=new_dilation,
        )

        self._residual_conv = nn.Conv2d(
            in_channels=conv_channels,
            out_channels=residual_channels,
            kernel_size=(1, 1),
        )

        if seq_length > receptive_field:
            self._skip_conv = nn.Conv2d(
                in_channels=conv_channels,
                out_channels=skip_channels,
                kernel_size=(1, seq_length - rf_size_j + 1),
            )
        else:
            self._skip_conv = nn.Conv2d(
                in_channels=conv_channels,
                out_channels=skip_channels,
                kernel_size=(1, receptive_field - rf_size_j + 1),
            )

        if gcn_true:
            self._mixprop_conv1 = MixProp(
                conv_channels, residual_channels, gcn_depth, dropout, propbeta, arquitecture, n_heads, concat
            )

            self._mixprop_conv2 = MixProp(
                conv_channels, residual_channels, gcn_depth, dropout, propbeta, arquitecture, n_heads, concat
            )

        if seq_length > receptive_field:
            self._normalization = LayerNormalization(
                (residual_channels, num_nodes, seq_length - rf_size_j + 1),
                elementwise_affine=layer_norm_affline,
            )

        else:
            self._normalization = LayerNormalization(
                (residual_channels, num_nodes, receptive_field - rf_size_j + 1),
                elementwise_affine=layer_norm_affline,
            )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self,
        X: torch.FloatTensor,
        X_skip: torch.FloatTensor,
        A_tilde: Optional[torch.FloatTensor],
        idx: torch.LongTensor,
        training: bool,
    ) -> torch.FloatTensor:
        """
        Making a forward pass of MTGNN layer.

        Arg types:
            * **X** (PyTorch FloatTensor) - Input feature tensor,
                with shape (batch_size, in_dim, num_nodes, seq_len).
            * **X_skip** (PyTorch FloatTensor) - Input feature tensor for skip connection,
                with shape (batch_size, in_dim, num_nodes, seq_len).
            * **A_tilde** (Pytorch FloatTensor or None) - Predefined adjacency matrix.
            * **idx** (Pytorch LongTensor) - Input indices.
            * **training** (bool) - Whether in traning mode.

        Return types:
            * **X** (PyTorch FloatTensor) - Output sequence tensor,
                with shape (batch_size, seq_len, num_nodes, seq_len).
            * **X_skip** (PyTorch FloatTensor) - Output feature tensor for skip connection,
                with shape (batch_size, in_dim, num_nodes, seq_len).
        """
        X_residual = X
        X_filter = self._filter_conv(X)
        X_filter = torch.tanh(X_filter)
        X_gate = self._gate_conv(X)
        X_gate = torch.sigmoid(X_gate)
        X = X_filter * X_gate
        X = F.dropout(X, self._dropout, training=training)
        X_skip = self._skip_conv(X) + X_skip
        if self._gcn_true:
            X = self._mixprop_conv1(X, A_tilde) + self._mixprop_conv2(
                X, A_tilde.transpose(1, 0)
            )
        else:
            X = self._residual_conv(X)

        X = X + X_residual[:, :, :, -X.size(3) :]
        X = self._normalization(X, idx)
        return X, X_skip


class MTGNNe(nn.Module):
    r"""An implementation of the Multivariate Time Series Forecasting Graph Neural Networks.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Extension of MTGNN

    Args:
        gcn_true (bool): Whether to add graph convolution layer.
        build_adj (bool): Whether to construct adaptive adjacency matrix.
        gcn_depth (int): Graph convolution depth.
        num_nodes (int): Number of nodes in the graph.
        kernel_set (list of int): List of kernel sizes.
        kernel_size (int): Size of kernel for convolution, to calculate receptive field size.
        dropout (float): Droupout rate.
        subgraph_size (int): Size of subgraph.
        node_dim (int): Dimension of nodes.
        dilation_exponential (int): Dilation exponential.
        conv_channels (int): Convolution channels.
        residual_channels (int): Residual channels.
        skip_channels (int): Skip channels.
        end_channels (int): End channels.
        seq_length (int): Length of input sequence.
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        layers (int): Number of layers.
        propbeta (float): Prop beta, ratio of retaining the root nodes's original states in mix-hop propagation, a value between 0 and 1.
        tanhalpha (float): Tanh alpha for generating adjacency matrix, alpha controls the saturation rate.
        layer_norm_affline (bool): Whether to do elementwise affine in Layer Normalization.
        xd (int, optional): Static feature dimension, default None.
        arquitecture (str): Type of arquitecture to use ('MTGNN'->original, 'MTGAN'->attention in adj_matrix, 'MTGNNAH'->MTGNN with attention in hops,'MTGANAH'->MTGAN with attention in hops)
        n_heads (int): Number of attention heads, in case to use MTGAN
        concat (bool): Whether the multi-head results should be concatenated or averaged, in case to use MTGAN
    """

    def __init__(
        self,
        gcn_true: bool,
        build_adj: bool,
        gcn_depth: int,
        num_nodes: int,
        kernel_set: list,
        kernel_size: int,
        dropout: float,
        subgraph_size: int,
        node_dim: int,
        dilation_exponential: int,
        conv_channels: int,
        residual_channels: int,
        skip_channels: int,
        end_channels: int,
        seq_length: int,
        in_dim: int,
        out_dim: int,
        layers: int,
        propbeta: float,
        tanhalpha: float,
        layer_norm_affline: bool,
        arquitecture: str,
        n_heads: int,
        concat: bool,
        xd: Optional[int] = None
    ):
        super(MTGNNe, self).__init__()

        self._gcn_true = gcn_true
        self._build_adj_true = build_adj
        self._num_nodes = num_nodes
        self._dropout = dropout
        self._seq_length = seq_length
        self._layers = layers
        self._idx = torch.arange(self._num_nodes)
        self.A = None # save the adjacency matrix

        self._mtgnn_layers = nn.ModuleList()

        self._graph_constructor = GraphConstructor(
            num_nodes, subgraph_size, node_dim, alpha=tanhalpha, xd=xd
        )

        self._set_receptive_field(dilation_exponential, kernel_size, layers)

        new_dilation = 1
        for j in range(1, layers + 1):
            self._mtgnn_layers.append(
                MTGNNLayer(
                    dilation_exponential=dilation_exponential,
                    rf_size_i=1,
                    kernel_size=kernel_size,
                    j=j,
                    residual_channels=residual_channels,
                    conv_channels=conv_channels,
                    skip_channels=skip_channels,
                    kernel_set=kernel_set,
                    new_dilation=new_dilation,
                    layer_norm_affline=layer_norm_affline,
                    gcn_true=gcn_true,
                    seq_length=seq_length,
                    receptive_field=self._receptive_field,
                    dropout=dropout,
                    gcn_depth=gcn_depth,
                    num_nodes=num_nodes,
                    propbeta=propbeta,
                    arquitecture=arquitecture,
                    n_heads=n_heads,
                    concat=concat
                )
            )

            new_dilation *= dilation_exponential

        self._setup_conv(
            in_dim, skip_channels, end_channels, residual_channels, out_dim
        )

        self._reset_parameters()

    def _setup_conv(
        self, in_dim, skip_channels, end_channels, residual_channels, out_dim
    ):

        self._start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1)
        )

        if self._seq_length > self._receptive_field:

            self._skip_conv_0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self._seq_length),
                bias=True,
            )

            self._skip_conv_E = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, self._seq_length - self._receptive_field + 1),
                bias=True,
            )

        else:
            self._skip_conv_0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self._receptive_field),
                bias=True,
            )

            self._skip_conv_E = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, 1),
                bias=True,
            )

        self._end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )

        self._end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True,
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def _set_receptive_field(self, dilation_exponential, kernel_size, layers):
        if dilation_exponential > 1:
            self._receptive_field = int(
                1
                + (kernel_size - 1)
                * (dilation_exponential ** layers - 1)
                / (dilation_exponential - 1)
            )
        else:
            self._receptive_field = layers * (kernel_size - 1) + 1

    def forward(
        self,
        X_in: torch.FloatTensor,
        A_tilde: Optional[torch.FloatTensor] = None,
        idx: Optional[torch.LongTensor] = None,
        FE: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass of MTGNN.

        Arg types:
            * **X_in** (PyTorch FloatTensor) - Input sequence, with shape (batch_size, in_dim, num_nodes, seq_len).
            * **A_tilde** (Pytorch FloatTensor, optional) - Predefined adjacency matrix, default None.
            * **idx** (Pytorch LongTensor, optional) - Input indices, a permutation of the num_nodes, default None (no permutation).
            * **FE** (Pytorch FloatTensor, optional) - Static feature, default None.

        Return types:
            * **X** (PyTorch FloatTensor) - Output sequence for prediction, with shape (batch_size, seq_len, num_nodes, 1).
        """
        seq_len = X_in.size(3)
        assert (
            seq_len == self._seq_length
        ), "Input sequence length not equal to preset sequence length."

        if self._seq_length < self._receptive_field:
            X_in = nn.functional.pad(
                X_in, (self._receptive_field - self._seq_length, 0, 0, 0)
            )

        if self._gcn_true:
            if self._build_adj_true:
                if idx is None:
                    A_tilde = self._graph_constructor(self._idx.to(X_in.device), FE=FE)
                else:
                    A_tilde = self._graph_constructor(idx, FE=FE)

        X = self._start_conv(X_in)
        X_skip = self._skip_conv_0(
            F.dropout(X_in, self._dropout, training=self.training)
        )
        if idx is None:
            for mtgnn in self._mtgnn_layers:
                X, X_skip = mtgnn(
                    X, X_skip, A_tilde, self._idx.to(X_in.device), self.training
                )
        else:
            for mtgnn in self._mtgnn_layers:
                X, X_skip = mtgnn(X, X_skip, A_tilde, idx, self.training)

        X_skip = self._skip_conv_E(X) + X_skip
        X = F.relu(X_skip)
        X = F.relu(self._end_conv_1(X))
        X = self._end_conv_2(X)
        self.A = A_tilde
        return X

class MTGNNe_Opt(nn.Module):
    r"""An implementation of the Multivariate Time Series Forecasting Graph Neural Networks.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Extension of MTGNN, and hyperparameter optimization configuration. The params with (opt) means that will be optimized

    Args:
        gcn_true (bool): Whether to add graph convolution layer.
        build_adj (bool): Whether to construct adaptive adjacency matrix.
        gcn_depth (int): Graph convolution depth. -> (opt)
        num_nodes (int): Number of nodes in the graph.
        kernel_set (list of int): List of kernel sizes.
        kernel_size (int): Size of kernel for convolution, to calculate receptive field size.
        dropout (float): Droupout rate. -> (opt)
        subgraph_size (int): Size of subgraph.
        node_dim (int): Dimension of nodes. -> (opt)
        dilation_exponential (int): Dilation exponential.
        conv_channels (int): Convolution channels.
        residual_channels (int): Residual channels.
        skip_channels (int): Skip channels.
        end_channels (int): End channels.
        seq_length (int): Length of input sequence. -> (opt)
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        layers (int): Number of layers. -> (opt)
        propbeta (float): Prop beta, ratio of retaining the root nodes's original states in mix-hop propagation, a value between 0 and 1. -> (opt)
        tanhalpha (float): Tanh alpha for generating adjacency matrix, alpha controls the saturation rate.
        layer_norm_affline (bool): Whether to do elementwise affine in Layer Normalization.
        xd (int, optional): Static feature dimension, default None.
        arquitecture (str): Type of arquitecture to use ('MTGNN'->original, 'MTGAN'->attention in adj_matrix, 
            'MTGNNAH'->MTGNN with attention in hops,'MTGANAH'->MTGAN with attention in hops) -> (opt)
        n_heads (int): Number of attention heads, in case to use MTGAN
        concat (bool): Whether the multi-head results should be concatenated or averaged, in case to use MTGAN
    """

    def __init__(
        self,
        trial,
        gcn_true: bool,
        build_adj: bool,
        #gcn_depth: int,
        num_nodes: int,
        kernel_set: list,
        kernel_size: int,
        #dropout: float,
        subgraph_size: int,
        #node_dim: int,
        dilation_exponential: int,
        conv_channels: int,
        residual_channels: int,
        skip_channels: int,
        end_channels: int,
        seq_length: int,
        in_dim: int,
        out_dim: int,
        #layers: int,
        #propbeta: float,
        tanhalpha: float,
        layer_norm_affline: bool,
        #arquitecture: str,
        n_heads: int,
        concat: bool,
        xd: Optional[int] = None
    ):
        super(MTGNNe_Opt, self).__init__()

        self._gcn_true = gcn_true
        self._build_adj_true = build_adj
        self._num_nodes = num_nodes
        
        self._idx = torch.arange(self._num_nodes)
        self.A = None # save the adjacency matrix

        gcn_depth = trial.suggest_int('gcn_depth', 1, 3, 1)
        dropout = trial.suggest_float('dropout', 0, 0.5, step = 0.1)
        node_dim = trial.suggest_int('node_dim', 20, 60, 20)
        layers = trial.suggest_int('layers', 1, 10, 1)
        propbeta = trial.suggest_float('propbeta', 0, 0.5, step=0.05)
        arquitecture = trial.suggest_categorical('arquitecture', ['MTGNN','MTGNNAH'])

        self._dropout = dropout
        self._seq_length = seq_length
        self._layers = layers

        self._mtgnn_layers = nn.ModuleList()

        self._graph_constructor = GraphConstructor(
            num_nodes, subgraph_size, node_dim, alpha=tanhalpha, xd=xd
        )

        self._set_receptive_field(dilation_exponential, kernel_size, layers)

        new_dilation = 1
        for j in range(1, layers + 1):
            self._mtgnn_layers.append(
                MTGNNLayer(
                    dilation_exponential=dilation_exponential,
                    rf_size_i=1,
                    kernel_size=kernel_size,
                    j=j,
                    residual_channels=residual_channels,
                    conv_channels=conv_channels,
                    skip_channels=skip_channels,
                    kernel_set=kernel_set,
                    new_dilation=new_dilation,
                    layer_norm_affline=layer_norm_affline,
                    gcn_true=gcn_true,
                    seq_length=seq_length,
                    receptive_field=self._receptive_field,
                    dropout=dropout,
                    gcn_depth=gcn_depth,
                    num_nodes=num_nodes,
                    propbeta=propbeta,
                    arquitecture=arquitecture,
                    n_heads=n_heads,
                    concat=concat
                )
            )

            new_dilation *= dilation_exponential

        self._setup_conv(
            in_dim, skip_channels, end_channels, residual_channels, out_dim
        )

        self._reset_parameters()

    def _setup_conv(
        self, in_dim, skip_channels, end_channels, residual_channels, out_dim
    ):

        self._start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1)
        )

        if self._seq_length > self._receptive_field:

            self._skip_conv_0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self._seq_length),
                bias=True,
            )

            self._skip_conv_E = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, self._seq_length - self._receptive_field + 1),
                bias=True,
            )

        else:
            self._skip_conv_0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self._receptive_field),
                bias=True,
            )

            self._skip_conv_E = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, 1),
                bias=True,
            )

        self._end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )

        self._end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True,
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def _set_receptive_field(self, dilation_exponential, kernel_size, layers):
        if dilation_exponential > 1:
            self._receptive_field = int(
                1
                + (kernel_size - 1)
                * (dilation_exponential ** layers - 1)
                / (dilation_exponential - 1)
            )
        else:
            self._receptive_field = layers * (kernel_size - 1) + 1

    def forward(
        self,
        X_in: torch.FloatTensor,
        A_tilde: Optional[torch.FloatTensor] = None,
        idx: Optional[torch.LongTensor] = None,
        FE: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass of MTGNN.

        Arg types:
            * **X_in** (PyTorch FloatTensor) - Input sequence, with shape (batch_size, in_dim, num_nodes, seq_len).
            * **A_tilde** (Pytorch FloatTensor, optional) - Predefined adjacency matrix, default None.
            * **idx** (Pytorch LongTensor, optional) - Input indices, a permutation of the num_nodes, default None (no permutation).
            * **FE** (Pytorch FloatTensor, optional) - Static feature, default None.

        Return types:
            * **X** (PyTorch FloatTensor) - Output sequence for prediction, with shape (batch_size, seq_len, num_nodes, 1).
        """
        seq_len = X_in.size(3)
        assert (
            seq_len == self._seq_length
        ), "Input sequence length not equal to preset sequence length."

        if self._seq_length < self._receptive_field:
            X_in = nn.functional.pad(
                X_in, (self._receptive_field - self._seq_length, 0, 0, 0)
            )

        if self._gcn_true:
            if self._build_adj_true:
                if idx is None:
                    A_tilde = self._graph_constructor(self._idx.to(X_in.device), FE=FE)
                else:
                    A_tilde = self._graph_constructor(idx, FE=FE)

        X = self._start_conv(X_in)
        X_skip = self._skip_conv_0(
            F.dropout(X_in, self._dropout, training=self.training)
        )
        if idx is None:
            for mtgnn in self._mtgnn_layers:
                X, X_skip = mtgnn(
                    X, X_skip, A_tilde, self._idx.to(X_in.device), self.training
                )
        else:
            for mtgnn in self._mtgnn_layers:
                X, X_skip = mtgnn(X, X_skip, A_tilde, idx, self.training)

        X_skip = self._skip_conv_E(X) + X_skip
        X = F.relu(X_skip)
        X = F.relu(self._end_conv_1(X))
        X = self._end_conv_2(X)
        self.A = A_tilde
        return X


class MTGNNe_Opt2(nn.Module):
    r"""An implementation of the Multivariate Time Series Forecasting Graph Neural Networks.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Extension of MTGNN, and hyperparameter optimization configuration. The params with (opt) means that will be optimized
    (opt*) refers that is an indirected optimization

    Args:
        gcn_true (bool): Whether to add graph convolution layer.
        build_adj (bool): Whether to construct adaptive adjacency matrix.
        gcn_depth (int): Graph convolution depth.
        num_nodes (int): Number of nodes in the graph.
        kernel_set (list of int): List of kernel sizes.
        kernel_size (int): Size of kernel for convolution, to calculate receptive field size.
        dropout (float): Droupout rate. -> (opt)
        subgraph_size (int): Size of subgraph.
        node_dim (int): Dimension of nodes.
        dilation_exponential (int): Dilation exponential.
        conv_channels (int): Convolution channels. -> (opt)
        residual_channels (int): Residual channels. -> (opt*)
        skip_channels (int): Skip channels. -> (opt*)
        end_channels (int): End channels. -> (opt*)
        seq_length (int): Length of input sequence. -> (opt)
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        layers (int): Number of layers. -> (opt)
        propbeta (float): Prop beta, ratio of retaining the root nodes's original states in mix-hop propagation, a value between 0 and 1. -> (opt)
        tanhalpha (float): Tanh alpha for generating adjacency matrix, alpha controls the saturation rate.
        layer_norm_affline (bool): Whether to do elementwise affine in Layer Normalization.
        xd (int, optional): Static feature dimension, default None.
        arquitecture (str): Type of arquitecture to use ('MTGNN'->original, 'MTGAN'->attention in adj_matrix, 
            'MTGNNAH'->MTGNN with attention in hops,'MTGANAH'->MTGAN with attention in hops)
        n_heads (int): Number of attention heads, in case to use MTGAN
        concat (bool): Whether the multi-head results should be concatenated or averaged, in case to use MTGAN
    """

    def __init__(
        self,
        trial,
        gcn_true: bool,
        build_adj: bool,
        gcn_depth: int,
        num_nodes: int,
        kernel_set: list,
        kernel_size: int,
        #dropout: float,
        subgraph_size: int,
        node_dim: int,
        dilation_exponential: int,
        conv_channels: int,
        residual_channels: int,
        skip_channels: int,
        end_channels: int,
        seq_length: int,
        in_dim: int,
        out_dim: int,
        #layers: int,
        #propbeta: float,
        tanhalpha: float,
        layer_norm_affline: bool,
        arquitecture: str,
        n_heads: int,
        concat: bool,
        xd: Optional[int] = None
    ):
        super(MTGNNe_Opt2, self).__init__()

        self._gcn_true = gcn_true
        self._build_adj_true = build_adj
        self._num_nodes = num_nodes
        
        self._idx = torch.arange(self._num_nodes)
        self.A = None # save the adjacency matrix

        dropout = trial.suggest_float('dropout', 0, 0.5, step = 0.1)
        conv_channels = trial.suggest_categorical('conv_channels', [8,16,32])
        # undirected optimization
        residual_channels = conv_channels
        skip_channels = conv_channels*2
        end_channels = skip_channels*2

        layers = trial.suggest_int('layers', 1, 10, 1)
        propbeta = trial.suggest_float('propbeta', 0, 0.5, step=0.05)

        self._dropout = dropout
        self._seq_length = seq_length
        self._layers = layers


        self._mtgnn_layers = nn.ModuleList()

        self._graph_constructor = GraphConstructor(
            num_nodes, subgraph_size, node_dim, alpha=tanhalpha, xd=xd
        )

        self._set_receptive_field(dilation_exponential, kernel_size, layers)

        new_dilation = 1
        for j in range(1, layers + 1):
            self._mtgnn_layers.append(
                MTGNNLayer(
                    dilation_exponential=dilation_exponential,
                    rf_size_i=1,
                    kernel_size=kernel_size,
                    j=j,
                    residual_channels=residual_channels,
                    conv_channels=conv_channels,
                    skip_channels=skip_channels,
                    kernel_set=kernel_set,
                    new_dilation=new_dilation,
                    layer_norm_affline=layer_norm_affline,
                    gcn_true=gcn_true,
                    seq_length=seq_length,
                    receptive_field=self._receptive_field,
                    dropout=dropout,
                    gcn_depth=gcn_depth,
                    num_nodes=num_nodes,
                    propbeta=propbeta,
                    arquitecture=arquitecture,
                    n_heads=n_heads,
                    concat=concat
                )
            )

            new_dilation *= dilation_exponential

        self._setup_conv(
            in_dim, skip_channels, end_channels, residual_channels, out_dim
        )

        self._reset_parameters()

    def _setup_conv(
        self, in_dim, skip_channels, end_channels, residual_channels, out_dim
    ):

        self._start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1)
        )

        if self._seq_length > self._receptive_field:

            self._skip_conv_0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self._seq_length),
                bias=True,
            )

            self._skip_conv_E = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, self._seq_length - self._receptive_field + 1),
                bias=True,
            )

        else:
            self._skip_conv_0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self._receptive_field),
                bias=True,
            )

            self._skip_conv_E = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, 1),
                bias=True,
            )

        self._end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )

        self._end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True,
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def _set_receptive_field(self, dilation_exponential, kernel_size, layers):
        if dilation_exponential > 1:
            self._receptive_field = int(
                1
                + (kernel_size - 1)
                * (dilation_exponential ** layers - 1)
                / (dilation_exponential - 1)
            )
        else:
            self._receptive_field = layers * (kernel_size - 1) + 1

    def forward(
        self,
        X_in: torch.FloatTensor,
        A_tilde: Optional[torch.FloatTensor] = None,
        idx: Optional[torch.LongTensor] = None,
        FE: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass of MTGNN.

        Arg types:
            * **X_in** (PyTorch FloatTensor) - Input sequence, with shape (batch_size, in_dim, num_nodes, seq_len).
            * **A_tilde** (Pytorch FloatTensor, optional) - Predefined adjacency matrix, default None.
            * **idx** (Pytorch LongTensor, optional) - Input indices, a permutation of the num_nodes, default None (no permutation).
            * **FE** (Pytorch FloatTensor, optional) - Static feature, default None.

        Return types:
            * **X** (PyTorch FloatTensor) - Output sequence for prediction, with shape (batch_size, seq_len, num_nodes, 1).
        """
        seq_len = X_in.size(3)
        assert (
            seq_len == self._seq_length
        ), "Input sequence length not equal to preset sequence length."

        if self._seq_length < self._receptive_field:
            X_in = nn.functional.pad(
                X_in, (self._receptive_field - self._seq_length, 0, 0, 0)
            )

        if self._gcn_true:
            if self._build_adj_true:
                if idx is None:
                    A_tilde = self._graph_constructor(self._idx.to(X_in.device), FE=FE)
                else:
                    A_tilde = self._graph_constructor(idx, FE=FE)

        X = self._start_conv(X_in)
        X_skip = self._skip_conv_0(
            F.dropout(X_in, self._dropout, training=self.training)
        )
        if idx is None:
            for mtgnn in self._mtgnn_layers:
                X, X_skip = mtgnn(
                    X, X_skip, A_tilde, self._idx.to(X_in.device), self.training
                )
        else:
            for mtgnn in self._mtgnn_layers:
                X, X_skip = mtgnn(X, X_skip, A_tilde, idx, self.training)

        X_skip = self._skip_conv_E(X) + X_skip
        X = F.relu(X_skip)
        X = F.relu(self._end_conv_1(X))
        X = self._end_conv_2(X)
        self.A = A_tilde
        return X