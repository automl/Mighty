from __future__ import annotations

import torch
from torch.autograd import grad
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from copy import deepcopy
from typing import TYPE_CHECKING, Callable, List
from mighty.mighty_runners.mighty_runner import MightyRunner

if TYPE_CHECKING:
    from omegaconf import DictConfig


class MightyMAMLRunner(MightyRunner):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.meta_params = self.agent.policy.parameters().copy()
        self.meta_optimizer = torch.optim.Adam(self.meta_params, lr=cfg.meta_lr)
        self.maml_epochs = cfg.maml_epochs
        # TODO: this should be a list of envs
        self.maml_tasks = cfg.maml_tasks

    def run(self):  # type: ignore
        all_rewards = []
        for maml_epoch in self.maml_epochs:
            iteration_loss = 0.0
            iteration_reward = 0.0
            for task in self.maml_tasks:
                self.agent.policy = self.meta_params.clone()
                # TODO: adjust steps
                self.train(env=task, num_steps=self.num_steps)

                # Compute Validation Loss
                evaluation_results = self.evaluate()
                # TODO: get loss from evaluation_results?
                # TODO: make sure loss here is NOT detached!
                loss = self.agent.policy_loss(evaluation_results)
                iteration_loss += loss
                iteration_reward += evaluation_results["mean_eval_reward"]

            # Print statistics
            print("\nIteration", maml_epoch)
            adaptation_reward = iteration_reward / len(self.maml_tasks)
            print("adaptation_reward", adaptation_reward)
            all_rewards.append(adaptation_reward)

            adaptation_loss = iteration_loss / len(self.maml_tasks)
            print("adaptation_loss", adaptation_loss.item())

            self.meta_optimizer.zero_grad()
            adaptation_loss.backward()
            self.meta_optimizer.step()


# Adapted from L2L library: https://github.com/learnables/learn2learn/tree/master
class MightyTRPOMAMLRunner(MightyRunner):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.meta_params = self.agent.policy.parameters().copy()
        self.meta_lr = cfg.meta_lr
        self.maml_epochs = cfg.maml_epochs
        # TODO: this should be a list of envs
        self.maml_tasks = cfg.maml_tasks

    def maml_update(self, model, lr, grads=None):  # type: ignore
        """
        [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py)

        **Description**

        Performs a MAML update on model using grads and lr.
        The function re-routes the Python object, thus avoiding in-place
        operations.

        NOTE: The model itself is updated in-place (no deepcopy), but the
            parameters' tensors are not.

        **Arguments**

        * **model** (Module) - The model to update.
        * **lr** (float) - The learning rate used to update the model.
        * **grads** (list, *optional*, default=None) - A list of gradients for each parameter
            of the model. If None, will use the gradients in .grad attributes.

        **Example**
        ~~~python
        maml = l2l.algorithms.MAML(Model(), lr=0.1)
        model = maml.clone() # The next two lines essentially implement model.adapt(loss)
        grads = autograd.grad(loss, model.parameters(), create_graph=True)
        maml_update(model, lr=0.1, grads)
        ~~~
        """
        if grads is not None:
            params = list(model.parameters())
            if not len(grads) == len(list(params)):
                msg = "WARNING:maml_update(): Parameters and gradients have different length. ("
                msg += str(len(params)) + " vs " + str(len(grads)) + ")"
                print(msg)
            for p, g in zip(params, grads):
                if g is not None:
                    p.update = -lr * g
        return self.update_module(model)

    def update_module(self, module, updates=None, memo=None):  # type: ignore
        r"""
        [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

        **Description**

        Updates the parameters of a module in-place, in a way that preserves differentiability.

        The parameters of the module are swapped with their update values, according to:
        \[
        p \gets p + u,
        \]
        where \(p\) is the parameter, and \(u\) is its corresponding update.


        **Arguments**

        * **module** (Module) - The module to update.
        * **updates** (list, *optional*, default=None) - A list of gradients for each parameter
            of the model. If None, will use the tensors in .update attributes.

        **Example**
        ~~~python
        error = loss(model(X), y)
        grads = torch.autograd.grad(
            error,
            model.parameters(),
            create_graph=True,
        )
        updates = [-lr * g for g in grads]
        l2l.update_module(model, updates=updates)
        ~~~
        """
        if memo is None:
            memo = {}
        if updates is not None:
            params = list(module.parameters())
            if not len(updates) == len(list(params)):
                msg = "WARNING:update_module(): Parameters and updates have different length. ("
                msg += str(len(params)) + " vs " + str(len(updates)) + ")"
                print(msg)
            for p, g in zip(params, updates):
                p.update = g

        # Update the params
        for param_key in module._parameters:
            p = module._parameters[param_key]
            if p in memo:
                module._parameters[param_key] = memo[p]
            else:
                if p is not None and hasattr(p, "update") and p.update is not None:
                    updated = p + p.update
                    p.update = None
                    memo[p] = updated
                    module._parameters[param_key] = updated

        # Second, handle the buffers if necessary
        for buffer_key in module._buffers:
            buff = module._buffers[buffer_key]
            if buff in memo:
                module._buffers[buffer_key] = memo[buff]
            else:
                if (
                    buff is not None
                    and hasattr(buff, "update")
                    and buff.update is not None
                ):
                    updated = buff + buff.update
                    buff.update = None
                    memo[buff] = updated
                    module._buffers[buffer_key] = updated

        # Then, recurse for each submodule
        for module_key in module._modules:
            module._modules[module_key] = self.update_module(
                module._modules[module_key],
                updates=None,
                memo=memo,
            )

        # Finally, rebuild the flattened parameters for RNNs
        # See this issue for more details:
        # https://github.com/learnables/learn2learn/issues/139
        if hasattr(module, "flatten_parameters"):
            module._apply(lambda x: x)
        return module

    def meta_surrogate_loss(  # type: ignore
        self, iteration_replays, iteration_policies: List, policy
    ) -> tuple:
        mean_loss = 0.0
        mean_kl = 0.0
        for task_replays, old_policy in zip(iteration_replays, iteration_policies):
            train_replays = task_replays[:-1]
            valid_episodes = task_replays[-1]
            self.agent.policy = self.clone_module(policy)  # type: ignore

            # Fast Adapt
            for train_episodes in train_replays:
                # TODO: this probably doesn't work out of the box
                self.agent.update(train_episodes)
            new_policy = self.clone_module(self.agent.policy)  # type: ignore

            # Useful values
            states = valid_episodes.state()
            actions = valid_episodes.action()

            # Compute KL
            old_densities = old_policy.density(states)
            new_densities = new_policy.density(states)
            kl = kl_divergence(new_densities, old_densities).mean()
            mean_kl += kl  # type: ignore

            # Compute Surrogate Loss
            # TODO: probably wrong key
            advantages = valid_episodes["advantages"]
            advantages = (
                (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            ).detach()
            old_log_probs = (
                old_densities.log_prob(actions).mean(dim=1, keepdim=True).detach()
            )
            new_log_probs = new_densities.log_prob(actions).mean(dim=1, keepdim=True)
            # TODO: is this accessible?
            mean_loss += self.agent.policy_loss(
                new_log_probs, old_log_probs, advantages
            )
        mean_kl /= len(iteration_replays)
        mean_loss /= len(iteration_replays)
        return mean_loss, mean_kl

    def hessian_vector_product(self, loss, parameters, damping=1e-5) -> Callable:  # type: ignore
        """
        [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/trpo.py)

        ## Description

        Returns a callable that computes the product of the Hessian of loss
        (w.r.t. parameters) with another vector, using Pearlmutter's trick.

        Note that parameters and the argument of the callable can be tensors
        or list of tensors.

        ## References

        1. Pearlmutter, B. A. 1994. “Fast Exact Multiplication by the Hessian.” Neural Computation.

        ## Arguments

        * `loss` (tensor) - The loss of which to compute the Hessian.
        * `parameters` (tensor or list) - The tensors to take the gradient with respect to.
        * `damping` (float, *optional*, default=1e-5) - Damping of the Hessian-vector product.

        ## Returns

        * `hvp(other)` (callable) - A function to compute the Hessian-vector product,
            given a vector or list `other`.
        """
        if not isinstance(parameters, torch.Tensor):
            parameters = list(parameters)
        grad_loss = grad(loss, parameters, create_graph=True, retain_graph=True)
        grad_loss = parameters_to_vector(grad_loss)  # type: ignore

        def hvp(other):  # type: ignore
            """
            TODO: The reshaping (if arguments are lists) is not efficiently implemented.
                  (It requires a copy) A good idea would be to have
                  vector_to_shapes(vec, shapes) or similar.

            NOTE: We can not compute the grads with a vector version of the parameters,
                  since that vector (created within the function) will be a different
                  tree that is was not used in the computation of the loss.
                  (i.e. you get 'One of the differentiated tensors was not used.')
            """
            shape = None
            if not isinstance(other, torch.Tensor):
                shape = [torch.zeros_like(o) for o in other]
                other = parameters_to_vector(other)
            grad_prod = torch.dot(grad_loss, other)
            hessian_prod = grad(grad_prod, parameters, retain_graph=True)
            hessian_prod = parameters_to_vector(hessian_prod)
            hessian_prod = hessian_prod + damping * other
            if shape is not None:
                vector_to_parameters(hessian_prod, shape)
                hessian_prod = shape
            return hessian_prod

        return hvp

    def conjugate_gradient(  # type: ignore
        self, Ax, b, num_iterations: int = 10, tol: float = 1e-10, eps: float = 1e-8
    ) -> torch.Tensor:
        """
        [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/trpo.py)

        ## Description

        Computes \\(x = A^{-1}b\\) using the conjugate gradient algorithm.

        ## Credit

        Adapted from Kai Arulkumaran's implementation, with additions inspired from John Schulman's implementation.

        ## References

        1. Nocedal and Wright. 2006. "Numerical Optimization, 2nd edition". Springer.
        2. Shewchuk et al. 1994. “An Introduction to the Conjugate Gradient Method without the Agonizing Pain.” CMU.

        ## Arguments

        * `Ax` (callable) - Given a vector x, computes A@x.
        * `b` (tensor or list) - The reference vector.
        * `num_iterations` (int, *optional*, default=10) - Number of conjugate gradient iterations.
        * `tol` (float, *optional*, default=1e-10) - Tolerance for proposed solution.
        * `eps` (float, *optional*, default=1e-8) - Numerical stability constant.

        ## Returns

        * `x` (tensor or list) - The solution to Ax = b, as a list if b is a list else a tensor.
        """
        shape = None
        if not isinstance(b, torch.Tensor):
            shape = [torch.zeros_like(b_i) for b_i in b]
            b = parameters_to_vector(b)
        x = torch.zeros_like(b)
        r = b
        p = r
        r_dot_old = torch.dot(r, r)
        for _ in range(num_iterations):
            Ap = Ax(p)
            alpha = r_dot_old / (torch.dot(p, Ap) + eps)
            x += alpha * p
            r -= alpha * Ap
            r_dot_new = torch.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
            if r_dot_new.item() < tol:
                break
        if shape is not None:
            vector_to_parameters(x, shape)
            x = shape  # type: ignore
        return x

    def run(self) -> tuple[dict, dict]:
        policy = deepcopy(self.agent.policy)
        for maml_epoch in self.maml_epochs:
            iteration_reward = 0.0
            iteration_replays = []
            iteration_policies = []
            for task in self.maml_tasks:
                for task in self.maml_tasks:
                    task_replay = []
                    self.agent.policy = deepcopy(policy)
                    # TODO: adjust steps
                    training_metrics = self.train(env=task, num_steps=self.num_steps)
                    clone = deepcopy(self.agent.policy)
                    task_replay.append(training_metrics)

                    # Compute Validation Loss
                    eval_results = self.evaluate()
                    task_replay.append(eval_results)
                    iteration_reward += eval_results["mean_eval_reward"]
                    iteration_replays.append(task_replay)
                    iteration_policies.append(clone)

                # Print statistics
                print("\nIteration", maml_epoch)
                adaptation_reward = iteration_reward / len(self.maml_tasks)
                print("adaptation_reward", adaptation_reward)

                # TRPO meta-optimization
                backtrack_factor = 0.5
                ls_max_steps = 15
                max_kl = 0.01

                # Compute CG step direction
                old_loss, old_kl = self.meta_surrogate_loss(
                    iteration_replays,
                    iteration_policies,
                    policy,
                )
                grad = grad(old_loss, policy.parameters(), retain_graph=True)
                grad = parameters_to_vector([g.detach() for g in grad])  # type: ignore
                Fvp = self.hessian_vector_product(old_kl, policy.parameters())
                step = self.conjugate_gradient(Fvp, grad)
                shs = 0.5 * torch.dot(step, Fvp(step))
                lagrange_multiplier = torch.sqrt(shs / max_kl)
                step = step / lagrange_multiplier
                step_ = [torch.zeros_like(p.data) for p in policy.parameters()]
                vector_to_parameters(step, step_)
                step = step_  # type: ignore
                del old_kl, Fvp, grad
                old_loss.detach_()

                # Line-search
                for ls_step in range(ls_max_steps):
                    stepsize = backtrack_factor**ls_step * self.meta_lr
                    clone = deepcopy(policy)
                    for p, u in zip(clone.parameters(), step):
                        p.data.add_(-stepsize, u.data)
                    new_loss, kl = self.meta_surrogate_loss(
                        iteration_replays,
                        iteration_policies,
                        clone,
                    )
                    if new_loss < old_loss and kl < max_kl:
                        for p, u in zip(policy.parameters(), step):
                            p.data.add_(-stepsize, u.data)
                        break
        eval_results = self.evaluate()
        self.close()
        # TODO: make pretty results
        train_results = {}  # type: ignore
        return train_results, eval_results
