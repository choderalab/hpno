import torch
import dgl
import hpno

def run():
    ds_tr = dgl.data.MiniGCDataset(100, 10, 20, seed=0)
    ds_vl = dgl.data.MiniGCDataset(100, 10, 20, seed=1)
    ds_te = dgl.data.MiniGCDataset(100, 10, 20, seed=2)

    g_tr, y_tr = zip(*ds_tr)
    g_vl, y_vl = zip(*ds_vl)
    g_te, y_te = zip(*ds_te)
    
    g_tr = [hpno.heterograph(g) for g in g_tr]
    g_vl = [hpno.heterograph(g) for g in g_vl]
    g_te = [hpno.heterograph(g) for g in g_te]

    from dgl.data.utils import save_graphs
    save_graphs("ds_tr.bin", g_tr, {"label": torch.stack(y_tr)})
    save_graphs("ds_vl.bin", g_vl, {"label": torch.stack(y_vl)})
    save_graphs("ds_te.bin", g_te, {"label": torch.stack(y_te)})


if __name__ == "__main__":
    run()
