# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl
import copy
import hpnotiq as hq

# =============================================================================
# MODULE CLASSES
# =============================================================================

class Train:
    """Training experiment.

    Parameters
    ----------
    net : `torch.nn.Module`
        Forward pass model that combines representation and output regression
        and generates predictive distribution.

    data : `List` of `tuple` of `(dgl.DGLGraph, torch.Tensor)`
        or `hpnotiq.data.dataset.Dataset`
        Pairs of graph, measurement.

    optimizer : `torch.optim.Optimizer`
        Optimizer for training.

    n_epochs : `int`
        Number of epochs.

    record_interval : `int`
        (Default value = 1)
        Interval states are recorded.

    lr_scheduler: `torch.optim.lr_scheduler`
        (Default value = None)
        Learning rate scheduler, will apply after every training epoch

    Methods
    -------
    train_once : Train model once.

    train : Train the model multiple times.


    """

    def __init__(self, net, data, optimizer, n_epochs=100, record_interval=1,
            loss_fn=hq.mse):

        self.data = data
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.net = net
        self.record_interval = record_interval
        self.loss_fn = loss_fn
        self.states = {}

    def train_once(self):
        """Train the model for one batch."""
        for g, y in self.data:

            def l():
                """ """
                self.optimizer.zero_grad()
                y_hat = self.net(g).nodes['g'].data['y_hat']
                loss = self.loss_fn(y, y_hat)
                loss.backward()
                return loss

            self.optimizer.step(l)

    def train(self):
        """Train the model for multiple steps and
        record the weights once every
        `record_interval`.

        Parameters
        ----------

        Returns
        -------

        """

        for epoch_idx in range(int(self.n_epochs)):
            self.train_once()

            if epoch_idx % self.record_interval == 0:
                self.states[epoch_idx] = copy.deepcopy(self.net.state_dict())

        self.states["final"] = copy.deepcopy(self.net.state_dict())

        if hasattr(self.optimizer, "expecation_params"):
            self.optimizer.expectation_params()

        return self.net


class Test:
    """ Test experiment. Metrics are applied to the saved states of the
    model to characterize its performance.


    Parameters
    ----------
    net : `hpnotiq.Net`
        Forward pass model that combines representation and output regression
        and generates predictive distribution.

    data : `List` of `tuple` of `(dgl.DGLGraph, torch.Tensor)`
        or `hpnotiq.data.dataset.Dataset`
        Pairs of graph, measurement.

    optimizer : `torch.optim.Optimizer`
        Optimizer for training.

    metrics : `List` of `callable`
        Metrics used to characterize the performance.

    Methods
    -------
    test : Run the test experiment.

    """

    def __init__(
        self, net, data, states, sampler=None, metrics=[hq.rmse, hq.r2]
    ):
        self.net = net  # deepcopy the model object
        self.data = data
        self.metrics = metrics
        self.states = states
        self.sampler = sampler

    def test(self):
        """ Run test experiment. """
        # switch to test
        self.net.eval()

        # initialize an empty dict for each metrics
        results = {}

        for metric in self.metrics:
            results[metric.__name__] = {}

        for state_name, state in self.states.items():  # loop through states
            self.net.load_state_dict(state)

            # concat y and y_hat in test set

            y = []
            g = []
            for g_, y_ in self.data:
                y.append(y_)
                if isinstance(g_, dgl.DGLGraph):
                    g.append(g_)
                else:
                    g += dgl.unbatch_hetero(g_)

            if y[0].dim() == 0:
                y = torch.stack(y)
            else:
                y = torch.cat(y)

            g = dgl.batch_hetero(g)

            y_hat = self.net(g).nodes['g'].data['y_hat']

            for metric in self.metrics:  # loop through the metrics
                results[metric.__name__][state_name] = (
                    metric(y, y_hat)
                    .detach()
                    .cpu()
                    .numpy()
                )

        self.results = results
        return dict(results)


class TrainAndTest:
    """ Run training and test experiment.

    Parameters
    ----------
    net : `hq.Net`
        Forward pass model that combines representation and output regression
        and generates predictive distribution.

    data : `List` of `tuple` of `(dgl.DGLGraph, torch.Tensor)`
        or `hq.data.dataset.Dataset`
        Pairs of graph, measurement.

    optimizer : `torch.optim.Optimizer` or `hq.Sampler`
        Optimizer for training.

    metrics : `List` of `callable`
        (Default value: `[hq.rmse, hq.r2, hq.pearsonr, hq.avg_nll]`)
        Metrics used to characterize the performance.

    n_epochs : `int`
        Number of epochs.

    record_interval : `int`
        (Default value = 1)
        Interval states are recorded.

    train_cls: `Train`
        (Default value = Train)
        Train class to use

    lr_scheduler: `torch.optim.lr_scheduler`
        (Default value = None)
        Learning rate scheduler, will apply after every training epoch

    Methods
    -------
    run : conduct experiment

    """

    def __init__(
        self,
        net,
        data_tr,
        data_te,
        optimizer,
        metrics=[hq.rmse, hq.r2,],
        n_epochs=100,
        record_interval=1,
        train_cls=Train,
        lr_scheduler=None,
    ):
        self.net = net  # deepcopy the model object
        self.data_tr = data_tr
        self.data_te = data_te
        self.metrics = metrics
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.record_interval = record_interval
        self.train_cls = train_cls
        self.lr_scheduler = lr_scheduler

    def __str__(self):
        _str = ""
        _str += "# model"
        _str += "\n"
        _str += str(self.net)
        _str += "\n"
        if hasattr(self.net, "noise_model"):
            _str += "# noise model"
            _str += "\n"
            _str += str(self.net.noise_model)
            _str += "\n"
        _str += "# optimizer"
        _str += "\n"
        _str += str(self.optimizer)
        _str += "\n"
        _str += "# n_epochs"
        _str += "\n"
        _str += str(self.n_epochs)
        _str += "\n"
        return _str

    def run(self):
        """ Run train and test experiments. """
        train = self.train_cls(
            net=self.net,
            data=self.data_tr,
            optimizer=self.optimizer,
            n_epochs=self.n_epochs,
            lr_scheduler=self.lr_scheduler
        )

        train.train()

        self.states = train.states

        test = Test(
            net=self.net,
            data=self.data_te,
            metrics=self.metrics,
            states=self.states,
            sampler=self.optimizer,
        )

        test.test()

        self.results_te = test.results

        test = Test(
            net=self.net,
            data=self.data_tr,
            metrics=self.metrics,
            states=self.states,
            sampler=self.optimizer,
        )

        test.test()

        self.results_tr = test.results

        del train

        return {"test": self.results_te, "training": self.results_tr}


class MultipleTrainAndTest:
    """A sequence of controlled experiment."""

    def __init__(self, experiment_generating_fn, param_dicts):
        self.experiment_generating_fn = experiment_generating_fn
        self.param_dicts = param_dicts

    def run(self):
        """ """
        results = []

        for param_dict in self.param_dicts:
            train_and_test = self.experiment_generating_fn(param_dict)
            result = train_and_test.run()
            results.append((param_dict, result))
            del train_and_test

        self.results = results

        return self.results
