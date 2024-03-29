import torch
import dgl
import hpno

def run(args):
    from ogb.graphproppred import DglGraphPropPredDataset, Evaluator, collate_dgl
    from torch.utils.data import DataLoader

    dataset = DglGraphPropPredDataset(name="ogbg-molhiv")

    import os
    if not os.path.exists("heterographs.bin"):
        dataset.graphs = [hpno.heterograph(graph) for graph in dataset.graphs]
        from dgl.data.utils import save_graphs
        save_graphs("heterographs.bin", dataset.graphs)
    else:
        from dgl.data.utils import load_graphs
        dataset.graphs = load_graphs("heterographs.bin")[0]

    evaluator = Evaluator(name="ogbg-molhiv")
    in_features = 9
    out_features = 1

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=128, drop_last=True, shuffle=True, collate_fn=collate_dgl)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=len(split_idx["valid"]), shuffle=False, collate_fn=collate_dgl)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=len(split_idx["test"]), shuffle=False, collate_fn=collate_dgl)

    model = hpno.HierarchicalPathNetwork(
        in_features=in_features,
        out_features=args.hidden_features,
        hidden_features=args.hidden_features,
        depth=args.depth,
        readout=hpno.GraphReadout(
            in_features=args.hidden_features,
            out_features=out_features,
            hidden_features=args.hidden_features,
        )
    )


    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=20)

    for idx_epoch in range(args.n_epochs):
        print(idx_epoch, flush=True)
        model.train()
        for g, y in train_loader:
            y = y.float()
            if torch.cuda.is_available():
                g = g.to("cuda:0")
                y = y.cuda()
            optimizer.zero_grad()
            y_hat = model.forward(g, g.nodes['n1'].data["feat"].float())
            loss = torch.nn.BCELoss()(
                input=y_hat.sigmoid(),
                target=y,
            )
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            g, y = next(iter(valid_loader))
            y = y.float()
            if torch.cuda.is_available():
                g = g.to("cuda:0")
                y = y.cuda()
            y_hat = model.forward(g, g.nodes['n1'].data["feat"].float())
            loss = torch.nn.BCELoss()(
                input=y_hat.sigmoid(),
                target=y,
            )
            scheduler.step(loss)

        if optimizer.param_groups[0]["lr"] <= 0.01 * args.learning_rate: break

    model = model.cpu()
    g, y = next(iter(valid_loader))
    rocauc_vl = evaluator.eval(
        {
            "y_true": y.float(),
            "y_pred": model.forward(g, g.nodes['n1'].data["feat"].float()).sigmoid()
        }
    )["rocauc"]

    g, y = next(iter(test_loader))
    rocauc_te = evaluator.eval(
        {
            "y_true": y.float(),
            "y_pred": model.forward(g, g.nodes['n1'].data["feat"].float()).sigmoid()
        }
    )["rocauc"]

    import pandas as pd
    df = pd.DataFrame(
        {
            args.data: {
                "rocauc_te": rocauc_te,
                "rocauc_vl": rocauc_vl,
            }
        }
    )

    df.to_csv("%s.csv" % args.out)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--data", type=str, default="cora")
    parser.add_argument("--hidden_features", type=int, default=16)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=4)
    args=parser.parse_args()
    run(args)
