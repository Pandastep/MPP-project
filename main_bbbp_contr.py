import os
# os.environ['CUDA_VISIBLE_DEVICES'] = f"0"
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from gnn_masking import GNN
from gru import SeqModel


from tqdm import tqdm
import argparse
import time
import numpy as np
import random
from datetime import datetime
from ogb.graphproppred import Evaluator
now = datetime.now()
timestamp = str(now.year)[-2:] + "_" + str(now.month).zfill(2) + "_" + str(now.day).zfill(2) + "_" + \
            str(now.hour).zfill(2) + str(now.minute).zfill(2) + str(now.second).zfill(2)

### importing OGB-LSC
cls_criterion = torch.nn.BCEWithLogitsLoss()

class FeatureReconstructionHead(torch.nn.Module):
    def __init__(self, emb_dim, out_dim):
        super(FeatureReconstructionHead, self).__init__()
        self.reconstruct = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, out_dim)
        )

    def forward(self, x):
        return self.reconstruct(x)


def mask_node_features_combined(batch, mask_ratio=0.3):
    node_feats = batch.x.clone()
    num_nodes, num_feats = node_feats.size()

    # Feature masking: 
    mask_node = torch.rand(num_nodes) < mask_ratio
    feat_mask = torch.rand_like(node_feats, dtype=torch.float32) < mask_ratio
    node_feats[mask_node] = node_feats[mask_node] * (~feat_mask[mask_node])

    # Node drop: 
    drop_node = torch.rand(num_nodes) < (mask_ratio / 2)
    node_feats[drop_node] = 0

    return node_feats, batch.x

def feature_mask_pretrain(model, train_loader, device, emb_dim, num_epochs=20, mask_ratio=0.3):
    model.train()
    recon_head = FeatureReconstructionHead(emb_dim=emb_dim, out_dim=9).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(recon_head.parameters()), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"[Pretrain Epoch {epoch+1}]"):
            batch = batch.to(device)

            if batch.x is None or batch.x.shape[0] == 0:
                print(">> Skipping empty batch")
                continue

            masked_x, original_x = mask_node_features_combined(batch, mask_ratio)
            batch.x = masked_x

            try:
                node_emb = model(batch, return_node_embedding=True)
                pred_feats = recon_head(node_emb)

                loss = loss_fn(pred_feats, original_x.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            except IndexError as e:
                print(f">> Skipping batch due to model error: {e}")
                continue

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"[Epoch {epoch+1}] Avg MSE Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "pretrained_masking.pt")
    print(">> Pretrained model saved.")

def train(model, device, loader, optimizer, evaluator):
    model.train()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:

            is_labeled = batch.y == batch.y
            pred = model(batch)
            optimizer.zero_grad()
            loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def test(model, device, loader):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1, )

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)

    return y_pred

from torch.nn import functional as F

def nt_xent_loss(z1, z2, temperature=0.5):
    """Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)"""
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.mm(representations, representations.T)

    sim_ij = torch.diag(similarity_matrix, len(z1))
    sim_ji = torch.diag(similarity_matrix, -len(z1))
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    nomask = torch.eye(len(representations), device=z1.device).bool()
    negatives = similarity_matrix[~nomask].view(len(representations), -1)

    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(len(logits), dtype=torch.long, device=z1.device)
    logits /= temperature
    return F.cross_entropy(logits, labels)


def train_multitask(model, device, loader, optimizer, evaluator, recon_head, recon_weight=1.0, class_weight=1.0, contrastive_weight=1.0):
    model.train()
    recon_head.train()
    loss_fn_recon = torch.nn.MSELoss()
    loss_fn_class = torch.nn.BCEWithLogitsLoss()

    y_true = []
    y_pred = []
    total_recon_loss = 0
    total_class_loss = 0
    total_contrastive_loss = 0

    for batch in tqdm(loader, desc="[Multitask Train]"):
        batch = batch.to(device)
        if batch.x.shape[0] <= 1:
            continue

        masked_x, original_x = mask_node_features_combined(batch, mask_ratio=0.3)
        batch.x = masked_x

        node_emb = model(batch, return_node_embedding=True)
        class_out = model(batch)
        recon_out = recon_head(node_emb)

        recon_loss = loss_fn_recon(recon_out, original_x.float())
        class_loss = loss_fn_class(class_out.float(), batch.y.float())
        contrastive_loss = nt_xent_loss(node_emb, node_emb)

        loss = recon_weight * recon_loss + class_weight * class_loss + contrastive_weight * contrastive_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_true.append(batch.y.view(class_out.shape).detach().cpu())
        y_pred.append(class_out.detach().cpu())
        total_recon_loss += recon_loss.item()
        total_class_loss += class_loss.item()
        total_contrastive_loss += contrastive_loss.item()

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    result = evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    result["recon_loss"] = total_recon_loss / len(loader)
    result["class_loss"] = total_class_loss / len(loader)
    result["contrastive_loss"] = total_contrastive_loss / len(loader)
    return result, result["rocauc"]


def eval_multitask(model, device, loader, evaluator, recon_head):
    model.eval()
    recon_head.eval()
    loss_fn_recon = torch.nn.MSELoss()
    loss_fn_class = torch.nn.BCEWithLogitsLoss()

    y_true = []
    y_pred = []
    total_recon_loss = 0
    total_class_loss = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Multitask Eval]"):
            batch = batch.to(device)
            if batch.x.shape[0] <= 1:
                continue

            masked_x, original_x = mask_node_features_combined(batch, mask_ratio=0.3)
            batch.x = masked_x

            node_emb = model(batch, return_node_embedding=True)
            class_out = model(batch)
            recon_out = recon_head(node_emb)

            recon_loss = loss_fn_recon(recon_out, original_x.float())
            class_loss = loss_fn_class(class_out.float(), batch.y.float())

            y_true.append(batch.y.view(class_out.shape).detach().cpu())
            y_pred.append(class_out.detach().cpu())
            total_recon_loss += recon_loss.item()
            total_class_loss += class_loss.item()

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    result = evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    result["recon_loss"] = total_recon_loss / len(loader)
    result["class_loss"] = total_class_loss / len(loader)
    return result, result["rocauc"]

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on pcqm4m with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=150,
                        help='dimensionality of hidden units in GNNs (default: 600)')
    parser.add_argument('--gru_emb', type=int, default=32,
                        help='GRU token embed size (default: 32)')
    parser.add_argument('--gru_hid', type=int, default=64,
                        help='GRU hidden size (default: 256)')
    parser.add_argument('--max_len', type=int, default=500,
                        help='')
    parser.add_argument('--train_subset', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--num_tasks', type=int, default=1,
                        help='num_labels, tox21: 12, PCQM4: 1')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default=f'ckpt/{timestamp}', help='directory to save checkpoint')
    parser.add_argument('--mask_ratio', type=float, default=0.3,
                    help='masking ratio for feature masking and node drop (default: 0.3)')
    args = parser.parse_args()

    print(args)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    train_dataset = torch.load("dataset/bbbp/processed/train_dataset.pt")
    valid_dataset = torch.load("dataset/bbbp/processed/valid_dataset.pt")
    test_dataset = torch.load("dataset/bbbp/processed/test_dataset.pt")

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator("ogbg-molbbbp")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.checkpoint_dir != '':
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    shared_params = {
        "num_tasks": args.num_tasks,
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling
    }

    if args.gnn == 'gin':
        model = GNN(gnn_type='gin', virtual_node=False, **shared_params).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type='gin', virtual_node=True, **shared_params).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type='gcn', virtual_node=False, **shared_params).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type='gcn', virtual_node=True, **shared_params).to(device)
    elif args.gnn == 'gru':
        model = SeqModel(args).to(device)
    else:
        raise ValueError('Invalid GNN type')

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')
 
    recon_head = FeatureReconstructionHead(emb_dim=args.emb_dim, out_dim=9).to(device)

    if args.log_dir != '':
        writer = SummaryWriter(log_dir=args.log_dir)

    print(">> Starting feature masking pretraining")
    feature_mask_pretrain(model, train_loader, device, emb_dim=args.emb_dim, num_epochs=20)

    print(">> Loading pretrained model")
    model.load_state_dict(torch.load("pretrained_masking.pt"))
 
 
    best_valid_auc = 0
    best_epoch = 0
     
    optimizer = optim.Adam(model.parameters(), lr=0.001)
 
    scheduler = StepLR(optimizer, step_size=30, gamma=0.25)
 
    for epoch in range(1, args.epochs + 1):
        print(f"===== Epoch {epoch} =====")

        train_result, train_auc = train_multitask(model, device, train_loader, optimizer, evaluator, recon_head)
        valid_result, valid_auc = eval_multitask(model, device, valid_loader, evaluator, recon_head)

        print({
            'Train ROC-AUC': train_auc,
            'Validation ROC-AUC': valid_auc,
            'Recon Loss': train_result["recon_loss"],
            'Class Loss': train_result["class_loss"],
            'Contrastive Loss': train_result["contrastive_loss"]
        })

        if args.log_dir != '':
            writer.add_scalar('valid/auc', valid_auc, epoch)
            writer.add_scalar('train/auc', train_auc, epoch)

        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_epoch = epoch
            if args.checkpoint_dir != '':
                print('Saving checkpoint')
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_auc': best_valid_auc,
                    'num_params': num_params
                }
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint.pt'))

            test_result, test_auc = eval_multitask(model, device, test_loader, evaluator, recon_head)
            print(f"Test ROC-AUC: {test_result['rocauc']}")

        scheduler.step()
        print(f'Best validation MAE so far: Epoch {best_epoch}: {best_valid_auc}')
    print(f"Final Test ROC-AUC: {test_result['rocauc']}")
    if args.log_dir != '':
        writer.close()


if __name__ == "__main__":
    main()

