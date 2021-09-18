import torch
import dgl
import hpno

def run(args):
    from dgl.data.utils import load_graphs
    ds_tr, y_tr = load_graphs("ds_tr.bin")
    ds_vl, y_vl = load_graphs("ds_vl.bin")
    ds_te, y_te = load_graphs("ds_te.bin")
    y_tr = y_tr["label"].float()
    y_vl = y_vl["label"].float()
    y_te = y_te["label"].float()
    g_tr = dgl.batch(ds_tr)
    g_vl = dgl.batch(ds_vl)
    g_te = dgl.batch(ds_te)

    in_features = 1
    out_features = 8

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
        y_tr = y_tr.cuda()
        y_vl = y_vl.cuda()
        y_te = y_te.cuda()
        g_tr = g_tr.to("cuda:0")
        g_vl = g_vl.to("cuda:0")
        g_te = g_te.to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=20)

    # dumb feature
    g_tr.nodes['n1'].data['h'] = torch.zeros(g_tr.number_of_nodes('n1'), 1, device=y_tr.device)
    g_vl.nodes['n1'].data['h'] = torch.zeros(g_vl.number_of_nodes('n1'), 1, device=y_tr.device)
    g_te.nodes['n1'].data['h'] = torch.zeros(g_te.number_of_nodes('n1'), 1, device=y_tr.device)

    for idx_epoch in range(args.n_epochs):
        print(idx_epoch, flush=True)
        model.train()
        optimizer.zero_grad()
        y_hat = model.forward(g_tr, g_tr.nodes['n1'].data["h"].float())
        loss = torch.nn.CrossEntropyLoss()(
            input=y_hat,
            target=y_tr.long(),
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_hat = model.forward(g_vl, g_vl.nodes['n1'].data["h"].float())
            loss = torch.nn.CrossEntropyLoss()(
                input=y_hat,
                target=y_vl.long(),
            )

            scheduler.step(loss)

        if optimizer.param_groups[0]["lr"] <= 0.01 * args.learning_rate: break


    accuracy_tr = (model(g_tr, g_tr.nodes['n1'].data['h'].float()).argmax(dim=-1) == y_tr).sum() / y_tr.shape[0]
    accuracy_te = (model(g_te, g_te.nodes['n1'].data['h'].float()).argmax(dim=-1) == y_te).sum() / y_te.shape[0]

    print(accuracy_tr)
    print(accuracy_te)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden_features", type=int, default=16)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--std", type=float, default=1.0)
    args=parser.parse_args()
    run(args)
