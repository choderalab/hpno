import pytest
import torch


def test_import():
    """ """
    import hpnotiq as hq
    import hpnotiq.app.experiment

def test_train():
    """ """
    import hpnotiq as hq

    net = torch.nn.Sequential(
        hq.models.input_layer.InputLayer(),
        hq.models.hmp.HierarchicalMessagePassing(),
        hq.models.output_layer.MultiPool(),
    )

    train = hq.Train(
        net=net,
        data=hq.data.esol()[:10],
        n_epochs=1,
        optimizer=torch.optim.Adam(net.parameters()),
    )

    train.train()

def test_test():
    """ """
    import hpnotiq as hq
    import copy

    net = torch.nn.Sequential(
        hq.models.input_layer.InputLayer(),
        hq.models.hmp.HierarchicalMessagePassing(),
        hq.models.output_layer.MultiPool(),
    )

    train = hq.Train(
        net=net,
        data=hq.data.utils.batch(hq.data.esol()[:10], 5),
        n_epochs=1,
        optimizer=torch.optim.Adam(net.parameters()),
    )

    train.train()

    test = hq.Test(
        net=net,
        data=hq.data.utils.batch(hq.data.esol()[:10], 5),
        metrics=[hq.mse, hq.rmse, hq.r2],
        states=train.states,
    )

    test.test()


def test_train_and_test():
    """ """
    import hpnotiq as hq

    net = torch.nn.Sequential(
        hq.models.input_layer.InputLayer(),
        hq.models.hmp.HierarchicalMessagePassing(),
        hq.models.output_layer.MultiPool(),
    )

    train_and_test = hq.TrainAndTest(
        net=net,
        optimizer=torch.optim.Adam(net.parameters(), 1e-3),
        n_epochs=1,
        data_tr=hq.data.utils.batch(hq.data.esol()[:10], 5),
        data_te=hq.data.utils.batch(hq.data.esol()[:10], 5),
    )

    print(train_and_test)
