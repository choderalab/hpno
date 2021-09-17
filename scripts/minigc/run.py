import torch
import dgl
import hpno

def run():
    ds_tr = dgl.data.MiniGCDataset(800, 10, 20)
    ds_vl = dgl.data.MiniGCDataset(100, 10, 20)
    ds_te = dgl.data.MiniGCDataset(100, 10, 20)

    g_tr, y_tr = zip(*ds_tr)
    g_vl, y_vl = zip(*ds_vl)
    g_te, y_te = zip(*ds_te)
    g_tr, y_tr = dgl.batch(g_tr), torch.stack(y_tr)
    g_vl, y_vl = dgl.batch(g_vl), torch.stack(y_vl)
    g_te, y_te = dgl.batch(g_te), torch.stack(y_te)

    g_tr = hpno.heterograph(g_tr)

if __name__ == "__main__":
    run()
