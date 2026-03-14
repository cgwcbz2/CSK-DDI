import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from itertools import chain
import argparse
import sys
import datetime
import numpy as np
import random
import json
from dataloader.data_process import rwExtractor
from dataloader.data_loader import DrugDataset
from model.feature_extractor import TripleChannelFeatureExtractor


# ================== 互注意力融合模块 ==================
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=128, num_heads=4, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.attn1 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn3 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, motif_emb, gnn_emb, kg_emb):
        motif = motif_emb.unsqueeze(1)
        gnn = gnn_emb.unsqueeze(1)
        kg = kg_emb.unsqueeze(1)
        motif_attn, _ = self.attn1(motif, gnn, gnn)
        motif_out = self.norm1(motif + motif_attn)
        gnn_attn, _ = self.attn2(gnn, kg, kg)
        gnn_out = self.norm2(gnn + gnn_attn)
        kg_attn, _ = self.attn3(kg, motif_out, motif_out)
        kg_out = self.norm3(kg + kg_attn)
        fused = (motif_out + gnn_out + kg_out) / 3
        fused = self.ffn(fused).squeeze(1)
        return fused


# ================== 融合分类模型 ==================
class FusionModel(nn.Module):
    def __init__(self, dim=128, hidden=512, num_classes=65):
        super(FusionModel, self).__init__()
        self.cross_fusion = CrossAttentionFusion(dim=dim, num_heads=4, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, motif_feat1, motif_feat2, g1_feat, g2_feat, kg1_feat, kg2_feat):
        drug1_emb = self.cross_fusion(motif_feat1, g1_feat, kg1_feat)
        drug2_emb = self.cross_fusion(motif_feat2, g2_feat, kg2_feat)

        x = torch.cat([drug1_emb, drug2_emb], dim=-1)
        logits = self.fc(x)
        return logits


# ================== argparse 参数 ==================
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument('--extractor', type=str, default="randomWalk")
parser.add_argument('--graph_fixed_num', type=int, default=1)
parser.add_argument('--khop', type=int, default=2)
parser.add_argument('--fixed_num', type=int, default=32)
parser.add_argument('--seed', type=int, default=42, help='随机种子')
args = parser.parse_args()

# ================== 设置随机种子 ==================
def set_seed(seed):
    """设置所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f" 已设置随机种子: {seed}")

set_seed(args.seed)

# ================== 日志 ==================
log_dir = "logs"
ckpt_dir = f"/root/autodl-tmp/check/checkpoints_lr{args.lr}_wd{args.weight_decay}_seed{args.seed}"
results_dir = f"results_lr{args.lr}_wd{args.weight_decay}_seed{args.seed}"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(log_dir, f"train_lr{args.lr}_seed{args.seed}_{timestamp}.txt")

class Logger(object):
    """同时打印并写入日志文件"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self. log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

sys.stdout = Logger(log_path)
print(f"日志文件: {log_path}")
print(f"超参数配置:")
print(f"  - Learning Rate: {args.lr}")
print(f"  - Seed:  {args.seed}")
print(f"  - Batch Size:  {args.batch_size}")
print(f"  - Epochs: {args.epochs}")
print("=" * 100)
print("开始训练融合模型...")
print("=" * 100 + "\n")

def collate_fn(batch):
    """
    批处理函数 - 分开 Motif，无分子特征
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None


    motif_seqs1, motif_seqs2, labels, h_data_list, t_data_list, kg1_list, kg2_list = zip(*batch)

    # Stack Motif 和 labels
    motif_seqs1 = torch.stack(motif_seqs1)  # [B, max_motif_len]
    motif_seqs2 = torch.stack(motif_seqs2)  # [B, max_motif_len]
    labels = torch.tensor(labels, dtype=torch.long)  # [B]

    # 批处理分子图
    h_batch = Batch.from_data_list(h_data_list)
    t_batch = Batch.from_data_list(t_data_list)

    # ============================================================
    # 处理 KG 子图
    # ============================================================
    # NUM_REL = 124
    NUM_REL = 126

    kg1_data_list = []
    kg2_data_list = []

    for i, (kg1, kg2) in enumerate(zip(kg1_list, kg2_list)):
        # 处理 kg1
        if kg1 is not None:
            try:
                # 兼容 7 或 8 元素
                if len(kg1) == 7:
                    node_idx, new_s_edge_index, new_s_rel, mapping_list, s_edge_index, s_value, s_rel = kg1
                else:
                    node_idx, new_s_edge_index, new_s_rel, mapping_list, s_edge_index, s_value, s_rel, degree = kg1

                # 过滤关系
                if len(s_rel) > 0:
                    s_rel = np.array(s_rel, dtype=np.int32)
                    valid_mask = s_rel < NUM_REL
                    s_rel = s_rel[valid_mask]
                    s_edge_index = np.array(s_edge_index)[valid_mask]
                    s_value = np.array(s_value)[valid_mask]

                if len(new_s_rel) > 0:
                    new_s_rel = np.array(new_s_rel, dtype=np.int32)
                    valid_mask = new_s_rel < NUM_REL
                    new_s_rel = new_s_rel[valid_mask]
                    new_s_edge_index = np.array(new_s_edge_index)[valid_mask]

                from torch_geometric.data import Data
                data1 = Data(
                    x=torch.LongTensor(node_idx),
                    edge_index=torch.LongTensor(new_s_edge_index).T if len(new_s_edge_index) > 0 else torch.empty(
                        (2, 0), dtype=torch.long),
                    id=torch.BoolTensor(mapping_list),
                    rel_index=torch.LongTensor(new_s_rel) if len(new_s_rel) > 0 else torch.empty(0, dtype=torch.long),
                    sp_edge_index=torch.LongTensor(s_edge_index).T if len(s_edge_index) > 0 else torch.empty((2, 0),
                                                                                                             dtype=torch.long),
                    sp_value=torch.FloatTensor(s_value) if len(s_value) > 0 else torch.empty(0, dtype=torch.float),
                    sp_edge_rel=torch.LongTensor(s_rel) if len(s_rel) > 0 else torch.empty(0, dtype=torch.long)
                )
                kg1_data_list.append(data1)
            except Exception as e:
                pass

        # 处理 kg2
        if kg2 is not None:
            try:
                if len(kg2) == 7:
                    node_idx, new_s_edge_index, new_s_rel, mapping_list, s_edge_index, s_value, s_rel = kg2
                else:
                    node_idx, new_s_edge_index, new_s_rel, mapping_list, s_edge_index, s_value, s_rel, degree = kg2

                # 过滤关系
                if len(s_rel) > 0:
                    s_rel = np.array(s_rel, dtype=np.int32)
                    valid_mask = s_rel < NUM_REL
                    s_rel = s_rel[valid_mask]
                    s_edge_index = np.array(s_edge_index)[valid_mask]
                    s_value = np.array(s_value)[valid_mask]

                if len(new_s_rel) > 0:
                    new_s_rel = np.array(new_s_rel, dtype=np.int32)
                    valid_mask = new_s_rel < NUM_REL
                    new_s_rel = new_s_rel[valid_mask]
                    new_s_edge_index = np.array(new_s_edge_index)[valid_mask]

                from torch_geometric.data import Data
                data2 = Data(
                    x=torch.LongTensor(node_idx),
                    edge_index=torch.LongTensor(new_s_edge_index).T if len(new_s_edge_index) > 0 else torch.empty(
                        (2, 0), dtype=torch.long),
                    id=torch.BoolTensor(mapping_list),
                    rel_index=torch.LongTensor(new_s_rel) if len(new_s_rel) > 0 else torch.empty(0, dtype=torch.long),
                    sp_edge_index=torch.LongTensor(s_edge_index).T if len(s_edge_index) > 0 else torch.empty((2, 0),
                                                                                                             dtype=torch.long),
                    sp_value=torch.FloatTensor(s_value) if len(s_value) > 0 else torch.empty(0, dtype=torch.float),
                    sp_edge_rel=torch.LongTensor(s_rel) if len(s_rel) > 0 else torch.empty(0, dtype=torch.long)
                )
                kg2_data_list.append(data2)
            except Exception as e:
                pass

    # 创建 Batch
    kg1_batch = Batch.from_data_list(kg1_data_list) if len(kg1_data_list) > 0 else None
    kg2_batch = Batch.from_data_list(kg2_data_list) if len(kg2_data_list) > 0 else None


    return motif_seqs1, motif_seqs2, labels, h_batch, t_batch, kg1_batch, kg2_batch


# ================== 数据加载 ==================
print("正在加载数据...")
train_dataset = DrugDataset("./datasets/Deng's_dataset/drug_smiles.csv", "./datasets/motif_vacb.csv", "training",
                             args=args)
val_dataset = DrugDataset("./datasets/Deng's_dataset/drug_smiles.csv", "./datasets/motif_vacb.csv", "validation",
                           args=args)
test_dataset = DrugDataset("./datasets/Deng's_dataset/drug_smiles.csv", "./datasets/motif_vacb.csv", "test", args=args)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

# ================== 模型初始化 ==================
print("正在初始化模型...")
device = args.device
num_classes = 65

print(f"\n KG 模型参数:")
print(f"  - num_nodes_kg: {train_dataset.num_node}")
print(f"  - num_rel_kg: {train_dataset.num_rel}")
print(f"  - max_degree_kg: {train_dataset.max_degree_kg}")

feature_extractor = TripleChannelFeatureExtractor(
    hidden_dim=args.hidden_dim,
    vocab_size=train_dataset.vocab_size,
    num_features_drug=len(train_dataset.smile_graph[str(train_dataset.ddi_pairs[0][0])][1][0]),
    gt_layer_num=4,
    gt_num_heads=8,
    num_rel_mol=int(train_dataset.num_rel_mol),
    max_degree_graph=int(train_dataset.max_smiles_degree),
    num_nodes_kg=train_dataset.num_node,
    num_rel_kg=train_dataset.num_rel,
    max_degree_kg=train_dataset.max_degree_kg
).to(device)

fusion_model = FusionModel(dim=128, hidden=512, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(chain(feature_extractor.parameters(), fusion_model.parameters()),
                             lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ================== 评估指标 ==================
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score


def compute_metrics(labels, predictions):
    precision = precision_score(labels, predictions, average="macro", zero_division=0)
    recall = recall_score(labels, predictions, average="macro", zero_division=0)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    kappa = cohen_kappa_score(labels, predictions)
    accuracy = sum([1 if p == l else 0 for p, l in zip(predictions, labels)]) / len(labels)
    return precision, recall, f1_macro, kappa, accuracy


# ================== 训练与验证循环 ==================
def run_epoch(loader, is_training=True):
    if is_training:
        feature_extractor.train()
        fusion_model.train()
    else:
        feature_extractor.eval()
        fusion_model.eval()

    total_loss = 0.0
    all_labels, all_preds = [], []

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch in loader:
            if batch is None:
                continue


            motif_seqs1, motif_seqs2, labels, h_batch, t_batch, kg1_batch, kg2_batch = batch


            motif_seqs1 = motif_seqs1.to(device)
            motif_seqs2 = motif_seqs2.to(device)
            labels = labels.to(device)
            h_batch = h_batch.to(device)
            t_batch = t_batch.to(device)

            if kg1_batch is not None:
                kg1_batch = kg1_batch.to(device)
            if kg2_batch is not None:
                kg2_batch = kg2_batch.to(device)

            if is_training:
                optimizer.zero_grad()

            motif_feat1, motif_feat2, d1_graph_feat, d2_graph_feat, h_kg_feat, t_kg_feat = feature_extractor(
                motif_seqs1, motif_seqs2, h_batch, t_batch, kg1_batch, kg2_batch
            )

            logits = fusion_model(motif_feat1, motif_feat2, d1_graph_feat, d2_graph_feat, h_kg_feat, t_kg_feat)

            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / (len(loader) if len(loader) > 0 else 1)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics, all_labels, all_preds


# ================== 主训练循环 ==================
def main_train_loop():
    best_f1, best_acc, best_epoch = 0, 0, 0
    print("开始训练模型.. .\n")
    for epoch in range(args.epochs):
        train_loss, train_metrics, _, _ = run_epoch(train_loader, is_training=True)
        val_loss, val_metrics, _, _ = run_epoch(val_loader, is_training=False)
        train_prec, train_rec, train_f1, train_kappa, train_acc = train_metrics
        val_prec, val_rec, val_f1, val_kappa, val_acc = val_metrics

        print(f"\n--- 轮次: {epoch + 1}/{args.epochs} ---")
        print(f"训练 - 损失: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f} | "
              f"Precision: {train_prec:.4f} | Recall: {train_rec:.4f} | Kappa: {train_kappa:.4f}")
        print(f"验证 - 损失: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | "
              f"Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | Kappa: {val_kappa:.4f}")

        if val_f1 > best_f1:
            best_f1, best_acc, best_epoch = val_f1, val_acc, epoch + 1
            print(f"[! ] 当前最佳模型: Epoch {best_epoch} | F1={best_f1:.4f}, Acc={best_acc:.4f}")
            ckpt_path = os.path.join(ckpt_dir, f"best_model_epoch{best_epoch}.pt")
            try:
                checkpoint = {
                    'epoch': epoch + 1,
                    'best_val_f1': best_f1,
                    'best_val_acc': best_acc,
                    'feature_extractor_state': feature_extractor.state_dict(),
                    'fusion_model_state': fusion_model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
                    'args': vars(args),
                    'torch_rng_state': torch.get_rng_state(),
                    'np_rng_state': np.random.get_state(),
                    'py_random_state': random.getstate()
                }
                if torch.cuda.is_available():
                    checkpoint['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
                torch.save(checkpoint, ckpt_path)
                print(f" 已保存最佳模型到: {ckpt_path}")
            except Exception as e:
                print(f" 保存 checkpoint 失败: {e}")

        scheduler.step(val_loss)

    print("\n训练完成")
    print(f"最佳模型出现在 Epoch {best_epoch}: F1={best_f1:.4f}, Acc={best_acc:.4f}")

    # 保存 Final 模型
    final_ckpt_path = os.path.join(ckpt_dir, f"final_model_epoch{args.epochs}.pt")
    try:
        final_checkpoint = {
            'epoch': args.epochs,
            'feature_extractor_state': feature_extractor.state_dict(),
            'fusion_model_state': fusion_model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'args': vars(args)
        }
        torch.save(final_checkpoint, final_ckpt_path)
        print(f" 已保存训练结束时的 final 模型到: {final_ckpt_path}")
    except Exception as e:
        print(f" 保存 final checkpoint 失败: {e}")

    # 测试 Best 模型
    print("\n" + "=" * 50)
    print(f"加载 Best 模型 (Epoch {best_epoch}) 进行测试...")
    print("=" * 50)

    best_ckpt_pattern = os.path.join(ckpt_dir, f"best_model_epoch{best_epoch}.pt")
    if os.path.exists(best_ckpt_pattern):
        print(f"加载最佳模型: {best_ckpt_pattern}")
        checkpoint = torch.load(best_ckpt_pattern, map_location=device)
        try:
            feature_extractor.load_state_dict(checkpoint['feature_extractor_state'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state'])
            print("✅ Best 模型参数已加载。")

            test_loss, test_metrics, test_labels, test_preds = run_epoch(test_loader, is_training=False)
            test_prec, test_rec, test_f1, test_kappa, test_acc = test_metrics

            print(f"\n[Best 模型] 测试结果 (Epoch {best_epoch}):")
            print(f"  损失: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
            print(f"  Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | Kappa: {test_kappa:.4f}")

            try:
                np.save(os.path.join(results_dir, f"test_labels_best_epoch{best_epoch}.npy"), np.array(test_labels))
                np.save(os.path.join(results_dir, f"test_preds_best_epoch{best_epoch}.npy"), np.array(test_preds))
                summary = {
                    'model_type': 'best_model',
                    'best_epoch': best_epoch,
                    'best_val_f1': float(best_f1),
                    'best_val_acc': float(best_acc),
                    'test_f1': float(test_f1),
                    'test_acc': float(test_acc),
                    'test_precision': float(test_prec),
                    'test_recall': float(test_rec),
                    'test_kappa': float(test_kappa)
                }
                with open(os.path.join(results_dir, f"test_summary_best_epoch{best_epoch}.json"), 'w',
                          encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                print(f"\n Best 模型的测试预测与 summary 已保存到: {results_dir}")
            except Exception as e:
                print(f" 保存 Best 模型的测试结果失败: {e}")

        except Exception as e:
            print(f" 无法完全加载 Best 模型的 state_dict: {e}")
    else:
        print(f"\n 未找到 Best 模型 checkpoint: {best_ckpt_pattern}，跳过测试。")

    # 测试 Final 模型
    print("\n" + "=" * 50)
    print(f"加载 Final 模型 (Epoch {args.epochs}) 进行测试...")
    print("=" * 50)

    final_ckpt_path = os.path.join(ckpt_dir, f"final_model_epoch{args.epochs}.pt")
    if os.path.exists(final_ckpt_path):
        try:
            ckpt = torch.load(final_ckpt_path, map_location=device)
            feature_extractor.load_state_dict(ckpt['feature_extractor_state'])
            fusion_model.load_state_dict(ckpt['fusion_model_state'])
            print(f"✅ Final 模型 ({final_ckpt_path}) 参数已加载。")

            final_loss, final_metrics, final_labels, final_preds = run_epoch(test_loader, is_training=False)
            final_prec, final_rec, final_f1, final_kappa, final_acc = final_metrics

            print(f"\n[Final 模型] 测试结果 (Epoch {args.epochs}):")
            print(f"  损失: {final_loss:.4f} | Acc: {final_acc:.4f} | F1: {final_f1:.4f}")
            print(f"  Precision: {final_prec:.4f} | Recall: {final_rec:.4f} | Kappa: {final_kappa:.4f}")

            try:
                np.save(os.path.join(results_dir, f"test_labels_final_epoch{args.epochs}.npy"), np.array(final_labels))
                np.save(os.path.join(results_dir, f"test_preds_final_epoch{args.epochs}.npy"), np.array(final_preds))
                final_summary = {
                    'model_type': 'final_model',
                    'final_epoch': args.epochs,
                    'test_f1': float(final_f1),
                    'test_acc': float(final_acc),
                    'test_precision': float(final_prec),
                    'test_recall': float(final_rec),
                    'test_kappa': float(final_kappa),
                    'ckpt_path': final_ckpt_path
                }
                with open(os.path.join(results_dir, f"test_summary_final_epoch{args.epochs}.json"), 'w',
                          encoding='utf-8') as f:
                    json.dump(final_summary, f, indent=2, ensure_ascii=False)
                print(f"\n Final 模型的测试预测与 summary 已保存到: {results_dir}")
            except Exception as e:
                print(f" 保存 Final 模型的测试结果失败: {e}")

        except Exception as e:
            print(f" 加载或测试 Final 模型失败: {e}")
    else:
        print(f" 未找到 Final 模型: {final_ckpt_path}，跳过测试。")


if __name__ == "__main__":
    main_train_loop()