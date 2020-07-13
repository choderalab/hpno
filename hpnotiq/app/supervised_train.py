import argparse
import hpnotiq as hq
import os
import numpy as np
import torch

def run(args):
    # define data
    data = getattr(
        hq.data,
        args.data)()

    # split
    partition = [int(x) for x in args.partition.split(":")]
    ds_tr, ds_te = hq.data.utils.split(data, partition)

    # batch
    ds_tr = hq.data.utils.batch(ds_tr, args.batch_size)
    ds_te = hq.data.utils.batch(ds_te, args.batch_size)


    net = torch.nn.Sequential(
        hq.models.input_layer.InputLayer(),
        hq.models.hmp.HierarchicalMessagePassing(),
        hq.models.output_layer.MultiPool(),
    )

    train_and_test = hq.TrainAndTest(
        net=net,
        optimizer=torch.optim.Adam(net.parameters(), args.lr),
        n_epochs=args.n_epochs,
        data_tr=ds_tr,
        data_te=ds_te,
    )

    print(hq.app.report.markdown(train_and_test.run()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--data', default='esol', type=str)
    parser.add_argument('--partition', default='4:1', type=str)
    parser.add_argument('--n_epochs', default=10, type=int)
    args = parser.parse_args()
    run(args)
