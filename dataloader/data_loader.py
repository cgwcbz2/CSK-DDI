# dataloader for motif sequence + molecular graph + KG subgraph
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from dataloader.data_process import smile_to_graph, load_entity2id, read_network, generate_node_subgraphs


class DrugDataset(Dataset):
    def __init__(self, drug_smiles_file, motif_seq_file, data_type, fold=0, ret_index=False, args=None,
                 max_motif_len=26, mask_prob=0.15):
        self.args = args
        self.max_motif_len = max_motif_len

        # 1. 加载药物SMILES
        df_smiles = pd.read_csv(drug_smiles_file)
        self.drug_smiles = dict(zip(df_smiles['drug_id'], df_smiles['smiles']))

        # 2. 加载DDI数据
        path = os.path.join("./datasets/Deng's_dataset/0/", f"ddi_{data_type}1xiao.csv")
        self.ddi_data = pd.read_csv(path)

        # 3. 加载预处理好的药物-Motif字典
        self.npz_data = np.load("datasets/drug_substructure_dict.npz", allow_pickle=True)

        # 4. 加载Motif序列数据和词汇表
        if motif_seq_file is not None and os.path.exists(motif_seq_file):
            self.motif_seq_data = pd.read_csv(motif_seq_file)
            self.vocab = self.load_vocab(motif_seq_file)
            self.vocab_size = len(self.vocab)
            print(f" 从文件加载 Motif 词汇表，大小:  {self.vocab_size}")
        else:
            self.motif_seq_data = None
            all_motifs = set()
            for drug_id in self.npz_data.files:
                motifs = self.npz_data[drug_id]
                all_motifs.update(motifs)
            self.vocab_size = len(all_motifs)
            print(f" 从 npz 构建 Motif 词汇表，大小: {self.vocab_size}")

        # 5. 加载KG相关数据（实体ID映射和KG边）
        self.entity2id = load_entity2id('datasets/entity2id.txt')
        network_path = 'datasets/networks.txt'
        num_node, network_edge_index, network_rel_index, num_rel = read_network(network_path)


        if isinstance(network_edge_index, np.ndarray) and len(network_edge_index) > 0:
            max_node_id = int(network_edge_index.max())
            if max_node_id >= num_node:
                print(f"调整 num_node:  {num_node} → {max_node_id + 1}")
                num_node = max_node_id + 1
        self.num_node = num_node
        self.num_rel = num_rel

        # drug_id_list 为 drug_smiles. csv 里所有药物映射到 KG 的整数实体 id
        drug_id_list = [
            self.entity2id.get(f"Compound: :{drug_id}", None)
            for drug_id in self.drug_smiles.keys()
        ]
        drug_id_list = [d for d in drug_id_list if d is not None]

        # 如果 args 为 None，创建默认配置
        if self.args is None:
            import argparse
            default_args = argparse.Namespace()
            default_args.extractor = 'extractor'
            default_args.num_hop = 2
        else:
            default_args = self.args


        subgraphs, max_degree, max_rel_num = generate_node_subgraphs(
            dataset="Deng's_dataset",
            drug_id=drug_id_list,
            network_edge_index=network_edge_index,
            network_rel_index=network_rel_index,
            num_rel=num_rel,
            args=default_args
        )


        self.max_degree_kg = int(max_degree) + 1  

        if max_rel_num >= self.num_rel:
            print(f" 调整 num_rel: {self.num_rel} → {max_rel_num + 1}")
            self.num_rel = max_rel_num + 1

        print(f" 子图统计:")
        print(f"  - max_degree: {max_degree}")
        print(f"  - max_degree_kg (模型使用): {self.max_degree_kg}")
        print(f"  - max_rel:  {max_rel_num}")
        print(f"  - num_rel (模型使用): {self.num_rel}")

        # 保存每个药物的KG子图
        self.kg_subgraphs = {}
        for drug_id in self.drug_smiles.keys():
            entity_id = self.entity2id.get(f"Compound::{drug_id}", None)
            if entity_id is not None and int(entity_id) in subgraphs:
                self.kg_subgraphs[drug_id] = subgraphs[int(entity_id)]
            else:
                self.kg_subgraphs[drug_id] = None
        valid_kg = sum(1 for v in self.kg_subgraphs.values() if v is not None)
        print(f" KG 子图:  {valid_kg}/{len(self.kg_subgraphs)} 个药物有有效子图")


        datapath = "./datasets/"
        print("Preprocessing molecular graphs...")
        self.smile_graph, self.num_rel_mol, self.max_smiles_degree = smile_to_graph(datapath, self.drug_smiles)
        self.num_rel_mol = self.num_rel_mol + 1
        self.max_smiles_degree = self.max_smiles_degree + 1
        print("Molecular graph preprocessing finished.")

        # --- 创建并过滤 DDI 对列表 ---
        raw_pairs = [
            (row['d1'], row['d2'], row['type'])
            for _, row in self.ddi_data.iterrows()
        ]

        def _norm(x):
            return str(x).strip()

        # 可用分子图与 KG 子图的药物集合
        smile_keys = set(_norm(k) for k in self.smile_graph.keys())
        kg_ok = set(_norm(d) for d, sub in self.kg_subgraphs.items() if sub is not None)
        entity_ok = set()
        for d in self.drug_smiles.keys():
            did = _norm(d)
            if f"Compound::{did}" in self.entity2id:
                entity_ok.add(did)

        valid_pairs = []
        skipped = []  # (d1, d2, type, reason)

        for d1, d2, typ in raw_pairs:
            k1, k2 = _norm(d1), _norm(d2)

            reason = None
            if f"Compound::{k1}" not in self.entity2id or f"Compound::{k2}" not in self.entity2id:
                reason = "no_entity2id"
            elif k1 not in smile_keys or k2 not in smile_keys:
                reason = "no_smile_graph"
            elif k1 not in kg_ok or k2 not in kg_ok:
                reason = "no_kg_subgraph"

            if reason is None:
                valid_pairs.append((d1, d2, typ))
            else:
                skipped.append((d1, d2, typ, reason))

        self.ddi_pairs = valid_pairs

        print(f" 最终用于训练/验证/测试的 DDI 对数:  {len(self.ddi_pairs)}")
        if skipped:
            os.makedirs("results", exist_ok=True)
            skip_path = os.path.join("results", "skipped_ddi_pairs_missing_kg.txt")
            with open(skip_path, "w", encoding="utf-8") as f:
                for a, b, t, r in skipped:
                    f.write(f"{a},{b},{t},{r}\n")
            # 汇总各原因数量
            from collections import Counter
            cnt = Counter(r for *_, r in skipped)
            print(f"跳过 {len(skipped)} 条 DDI 对（原因分布:  {dict(cnt)}），详情见 {skip_path}")
        else:
            print("所有 DDI 对均具备 entity2id、分子图与 KG 子图。")

    def load_vocab(self, file_path):
        vocab = {}
        data = pd.read_csv(file_path)
        for _, row in data.iterrows():
            vocab[row['order_num']] = row['motif_id']
        return vocab

    def __getitem__(self, index):
        # 获取药物对信息和标签
        drug_id1, drug_id2, label = self.ddi_pairs[index]
        label = torch.tensor(label, dtype=torch.long)

        try:
            h_motifs = self.npz_data[str(drug_id1)]
            t_motifs = self.npz_data[str(drug_id2)]
        except (KeyError, IndexError):
            print(f"Missing motifs for drug_id1: {drug_id1} or drug_id2: {drug_id2}")
            return None, None, None, None, None, None, None

        def pad_motif(motifs):
            motifs_list = list(motifs)
            if len(motifs_list) >= self.max_motif_len:
                return motifs_list[:self.max_motif_len]
            else:
                return motifs_list + [self.vocab_size] * (self.max_motif_len - len(motifs_list))

        motif_seq1 = torch.tensor(pad_motif(h_motifs), dtype=torch.long)
        motif_seq2 = torch.tensor(pad_motif(t_motifs), dtype=torch.long)

        # 药物1的分子图
        graph_info1 = self.smile_graph[str(drug_id1)]
        c_size1, features1, edge_index1, rel_index1, sp_edge_index1, sp_value1, sp_rel1, _ = graph_info1
        h_data = Data(
            x=torch.tensor(np.array(features1), dtype=torch.float),
            edge_index=torch.LongTensor(edge_index1).t().contiguous(),
            y=label,
            sp_edge_index=torch.LongTensor(sp_edge_index1).t().contiguous(),
            sp_value=torch.Tensor(np.array(sp_value1, dtype=int)),
            sp_edge_rel=torch.LongTensor(np.array(sp_rel1, dtype=int)),
            c_size=torch.LongTensor([c_size1])
        )
        h_data.drug_id = str(drug_id1)

        # 药物2的分子图
        graph_info2 = self.smile_graph[str(drug_id2)]
        c_size2, features2, edge_index2, rel_index2, sp_edge_index2, sp_value2, sp_rel2, _ = graph_info2
        t_data = Data(
            x=torch.tensor(np.array(features2), dtype=torch.float),
            edge_index=torch.LongTensor(edge_index2).t().contiguous(),
            y=label,
            sp_edge_index=torch.LongTensor(sp_edge_index2).t().contiguous(),
            sp_value=torch.Tensor(np.array(sp_value2, dtype=int)),
            sp_edge_rel=torch.LongTensor(np.array(sp_rel2, dtype=int)),
            c_size=torch.LongTensor([c_size2])
        )
        t_data.drug_id = str(drug_id2)

        # KG
        kg_subgraph1 = self.kg_subgraphs.get(drug_id1, None)
        kg_subgraph2 = self.kg_subgraphs.get(drug_id2, None)

        return motif_seq1, motif_seq2, label, h_data, t_data, kg_subgraph1, kg_subgraph2

    def __len__(self):
        return len(self.ddi_pairs)