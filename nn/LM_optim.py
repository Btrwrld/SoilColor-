import torch
from functools import reduce
from torch.optim import Optimizer


class LM(Optimizer):
    """Implements Levenberg-Marquad algorithm, heavily inspired by 

    .. warning::
        This optimizer is developed by me 

    Arguments:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """

    def __init__(   self,
                    params,
                    tau=1e-2,
                    eps1=1e-6,
                    eps2=1e-6,
                    max_iter=20,
                    using_qr=True):

        # TODO IMPLEMENT VALUE CHECKS FOR ALL THE ARGUMENTS


        defaults = dict(
                        tau=tau,
                        eps1=eps1,
                        eps2=eps2,
                        max_iter=max_iter,
                        using_qr=using_qr)
        super(LM, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LM doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None


    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)



    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """

        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        tau = group['tau']
        eps1 = group['eps1']
        eps2 = group['eps2']
        max_iter = group['max_iter']
        using_qr = group['using_qr']

        # Set states
        state = self.state[self._params[0]]
        state.setdefault('n_iter', 0)

        # Evaluate initial f(x) and df/dx
        loss = float(closure())
        prev_loss = loss

        flat_grad = self._gather_flat_grad()

        n_iter = 0
        print(10*'-')
        print(type(self._params))
        print(10*'-')
        print(self._params)
                






