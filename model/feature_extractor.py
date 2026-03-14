# feature_extractor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from model.tiger import NodeFeatures
from model.GraphTransformer import GraphTransformer


class ResidualLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.adjust_dim = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x):
        out = F.relu(self.fc(x))
        residual = self.adjust_dim(x) if self.adjust_dim is not None else x
        return out + residual


class TripleChannelFeatureExtractor(nn.Module):
    """
    motif + 分子图 + KG 三通道特征抽取器
    """
    def __init__(self, hidden_dim, vocab_size,
                 num_features_drug, gt_layer_num, gt_num_heads,
                 num_rel_mol, max_degree_graph,
                 num_nodes_kg, num_rel_kg, max_degree_kg):
        super(TripleChannelFeatureExtractor, self).__init__()

        # ============================================================
        # 1) Motif 分支
        # ============================================================
        self.motif_embedding = nn.Embedding(vocab_size + 1, hidden_dim, padding_idx=vocab_size)
        self.motif_seq_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=6
        )
        self.motif_to_128d = ResidualLayer(hidden_dim, 128)

        # ============================================================
        # 2) 分子图分支
        # ============================================================
        self.mol_atom_feature_encoder = NodeFeatures(
            degree=max_degree_graph,
            feature_num=num_features_drug,
            embedding_dim=hidden_dim,
            type='graph'
        )
        self.mol_representation_learning = GraphTransformer(
            layer_num=gt_layer_num,
            embedding_dim=hidden_dim,
            num_heads=gt_num_heads,
            num_rel=num_rel_mol,
            type='graph'
        )
        self.mol_to_128d = ResidualLayer(hidden_dim, 128)

        # ============================================================
        # 3) KG 分支
        # ============================================================
        self.kg_node_feature = NodeFeatures(
            degree=max_degree_kg,
            feature_num=num_nodes_kg,
            embedding_dim=hidden_dim,
            type='node'
        )
        self.kg_transformer = GraphTransformer(
            layer_num=gt_layer_num,
            embedding_dim=hidden_dim,
            num_heads=gt_num_heads,
            num_rel=num_rel_kg,
            type='node'
        )
        self.kg_to_128d = ResidualLayer(hidden_dim, 128)

    def encode_motif(self, motif_seq):
        if motif_seq.dim() == 1:
            motif_seq = motif_seq.unsqueeze(0)

        motif_embed = self.motif_embedding(motif_seq)  # [B, L, H]
        transformer_out = self.motif_seq_encoder(motif_embed)  # [B, L, H]
        motif_feature_hidden = torch.mean(transformer_out, dim=1)  # [B, H]
        motif_feature_128d = self.motif_to_128d(motif_feature_hidden)  # [B, 128]

        return motif_feature_128d

    def encode_kg(self, kg_batch, return_attention=False):
        device = next(self.parameters()).device

        if kg_batch is None:
            if return_attention:
                return torch.zeros(1, 128, device=device), None
            return torch.zeros(1, 128, device=device)

        node_feature = self.kg_node_feature(kg_batch)

        graph_emb, sub_representation, attn_layer = self.kg_transformer(node_feature, kg_batch)

        kg_feature_128d = self.kg_to_128d(graph_emb)

        if return_attention:
            # 返回注意力信息
            kg_attention = {
                'attn_layer': attn_layer,  # 每一层的注意力权重
                'sub_representation': sub_representation,  # 子图节点表示
                'node_feature': node_feature  # 节点特征
            }
            return kg_feature_128d, kg_attention

        return kg_feature_128d

    def forward(self, motif_seq1, motif_seq2, h_data, t_data, kg_batch1, kg_batch2, return_kg_attention=False):
        """
        三通道特征提取
        """
        # Motif
        motif_feature1 = self.encode_motif(motif_seq1)
        motif_feature2 = self.encode_motif(motif_seq2)

        # MolGraph
        h_node_feature = self.mol_atom_feature_encoder(h_data)
        h_graph_embedding, _, _ = self.mol_representation_learning(h_node_feature, h_data)
        h_graph_embedding = self.mol_to_128d(h_graph_embedding)

        t_node_feature = self.mol_atom_feature_encoder(t_data)
        t_graph_embedding, _, _ = self.mol_representation_learning(t_node_feature, t_data)
        t_graph_embedding = self.mol_to_128d(t_graph_embedding)

        # KG
        if return_kg_attention:
            h_kg_embedding, h_kg_attn = self.encode_kg(kg_batch1, return_attention=True)
            t_kg_embedding, t_kg_attn = self.encode_kg(kg_batch2, return_attention=True)

            return (motif_feature1, motif_feature2, h_graph_embedding, t_graph_embedding,
                    h_kg_embedding, t_kg_embedding, h_kg_attn, t_kg_attn)
        else:
            h_kg_embedding = self.encode_kg(kg_batch1)
            t_kg_embedding = self.encode_kg(kg_batch2)

            return (motif_feature1, motif_feature2, h_graph_embedding, t_graph_embedding,
                    h_kg_embedding, t_kg_embedding)