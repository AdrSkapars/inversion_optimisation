import torch

class DefaultAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(DefaultAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # First moment (m_t)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # Second moment (v_t)

                m, v = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                t = state['step']

                m.mul_(beta1).add_(grad, alpha=1 - beta1) # m_t = β1 * m_{t-1} + (1 - β1) * g_t
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) # v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
                m_hat = m / (1 - beta1**t) # m̂_t = m_t / (1 - β1^t)
                v_hat = v / (1 - beta2**t) # v̂_t = v_t / (1 - β2^t)
                denom = v_hat.sqrt().add(group['eps'])
                p.data.addcdiv_(m_hat, denom, value=-group['lr']) # θ_t = θ_{t-1} - η * m̂_t / (sqrt(v̂_t) + ε)

        return loss


class CustomAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(CustomAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)  # First moment (m_t)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # Second moment (v_t)

                m, v = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                m.mul_(beta1).add_(grad, alpha=1 - beta1) # m_t = β1 * m_{t-1} + (1 - β1) * g_t
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) # v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
                m_hat = m # m̂_t = m_t
                v_hat = v # v̂_t = v_t
                denom = v_hat.sqrt().add(group['eps'])
                p.data.addcdiv_(m_hat, denom, value=-group['lr']) # θ_t = θ_{t-1} - η * m̂_t / (sqrt(v̂_t) + ε)

        return loss
    

class LIONAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9):
        defaults = dict(lr=lr, beta1=beta1)
        super(LIONAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)  # First moment (m_t)

                m = state['exp_avg']
                beta1 = group['beta1']

                m.mul_(beta1).add_(grad, alpha=1 - beta1) # m_t = β1 * m_{t-1} + (1 - β1) * g_t
                m_hat = m # m̂_t = m_t
                p.data.add_(m_hat.sign(), alpha=-group['lr']) # θ_t = θ_{t-1} - η * sign(m̂_t)

        return loss