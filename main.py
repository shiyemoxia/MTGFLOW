#%%
import os
import csv
import argparse
import torch
from models.MTGFLOW import MTGFLOW
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve 

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, 
                    default='Data/input/SWaT_Dataset_Attack_v0.csv', help='Location of datasets.')
parser.add_argument('--output_dir', type=str, 
                    default='./checkpoint/')
parser.add_argument('--name',default='SWaT', help='the name of dataset')

parser.add_argument('--graph', type=str, default='None')
parser.add_argument('--model', type=str, default='MAF')


parser.add_argument('--n_blocks', type=int, default=1, help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--n_components', type=int, default=1, help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
parser.add_argument('--input_size', type=int, default=1)
parser.add_argument('--batch_norm', type=bool, default=False)
parser.add_argument('--train_split', type=float, default=0.6)
parser.add_argument('--stride_size', type=int, default=10)

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--window_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate.')
parser.add_argument('--epochs', type=int, default=40, help='Training epochs per seed.')
parser.add_argument('--seed_start', type=int, default=15, help='Inclusive seed start.')
parser.add_argument('--seed_end', type=int, default=19, help='Inclusive seed end.')
parser.add_argument('--setting', type=str, default='unsupervised', choices=['unsupervised', 'occ'],
                    help='Dataset split setting.')



args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


def build_loaders(args):
    from Dataset import (
        load_smd_smap_msl,
        load_smd_smap_msl_occ,
        loader_PSM,
        loader_PSM_OCC,
        loader_SWat,
        loader_SWat_OCC,
        loader_WADI,
        loader_WADI_OCC,
    )

    use_occ = args.setting == 'occ'
    name = str(args.name)
    name_lower = name.lower()

    if name_lower == 'swat':
        loader = loader_SWat_OCC if use_occ else loader_SWat
        return loader(args.data_dir, args.batch_size, args.window_size, args.stride_size, args.train_split)

    if name_lower == 'wadi':
        loader = loader_WADI_OCC if use_occ else loader_WADI
        return loader(args.data_dir, args.batch_size, args.window_size, args.stride_size, args.train_split)

    if name_lower == 'psm':
        loader = loader_PSM_OCC if use_occ else loader_PSM
        return loader(args.data_dir, args.batch_size, args.window_size, args.stride_size, args.train_split)

    if name == 'SMAP' or name == 'MSL' or name.startswith('machine'):
        loader = load_smd_smap_msl_occ if use_occ else load_smd_smap_msl
        return loader(name, args.batch_size, args.window_size, args.stride_size, args.train_split, root=args.data_dir)

    raise ValueError(f"Unsupported dataset name: {args.name}")
save_name = f"{args.name}_{args.setting}" if args.setting != 'unsupervised' else args.name
save_root = os.path.join(args.output_dir, save_name)
os.makedirs(save_root, exist_ok=True)
seed_rows = []

for seed in range(args.seed_start, args.seed_end + 1):
    args.seed = seed
    print(args)
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    #%%
    print("Loading dataset")
    print(args.name)
    train_loader, val_loader, test_loader, n_sensor = build_loaders(args)



    #%%
    model = MTGFLOW(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.window_size, n_sensor, dropout=0.0, model = args.model, batch_norm=args.batch_norm)
    model = model.to(device)

    #%%
    from torch.nn.utils import clip_grad_value_
    loss_best = 100
    roc_max = 0
    best_checkpoint_path = os.path.join(save_root, f"model_seed{seed}.pth")
    legacy_checkpoint_path = os.path.join(save_root, "model.pth")
  
    lr = args.lr 
    optimizer = torch.optim.Adam([
        {'params':model.parameters(), 'weight_decay':args.weight_decay},
        ], lr=lr, weight_decay=0.0)

    for epoch in range(args.epochs):
        print(epoch)
        loss_train = []

        model.train()
        for x,_,idx in train_loader:
            x = x.to(device)

            optimizer.zero_grad()
            loss = -model(x,)

            total_loss = loss

            total_loss.backward()
            clip_grad_value_(model.parameters(), 1)
            optimizer.step()
            loss_train.append(loss.item())



        loss_test = []
        with torch.no_grad():
            for x,_,idx in test_loader:

                x = x.to(device)
                loss = -model.test(x, ).cpu().numpy()
                loss_test.append(loss)
        loss_test = np.concatenate(loss_test)

    
        roc_test = roc_auc_score(np.asarray(test_loader.dataset.label,dtype=int),loss_test)

    
        if roc_max < roc_test:
            roc_max = roc_test
            torch.save({
            'model': model.state_dict(),
            }, best_checkpoint_path)
            torch.save({
            'model': model.state_dict(),
            }, legacy_checkpoint_path)

        roc_max = max(roc_test, roc_max)
        print(roc_max)

    print(f"Best ROC for seed {seed}: {roc_max:.10f}")
    seed_rows.append(
        {
            "seed": seed,
            "best_roc_auc": roc_max,
            "checkpoint_path": best_checkpoint_path,
        }
    )

results_csv_path = os.path.join(save_root, "seed_results.csv")
with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["seed", "best_roc_auc", "checkpoint_path"])
    writer.writeheader()
    writer.writerows(seed_rows)

if seed_rows:
    values = np.asarray([row["best_roc_auc"] for row in seed_rows], dtype=np.float64)
    print("==== Seed Summary ====")
    for row in seed_rows:
        print(
            f"seed={row['seed']:>4} | best_roc_auc={row['best_roc_auc']:.4f} | "
            f"checkpoint={row['checkpoint_path']}"
        )
    print(
        f"mean={values.mean():.4f} std={values.std(ddof=0):.4f} "
        f"min={values.min():.4f} max={values.max():.4f}"
    )
    print(f"Results CSV saved to: {results_csv_path}")
