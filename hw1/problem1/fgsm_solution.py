"""Problem 1 attacks using the network and seed from the course fgsm.py."""

import torch
from torch import nn


def make_problem():
    torch.manual_seed(13)
    model = nn.Sequential(
        nn.Linear(10, 10, bias=False),
        nn.ReLU(),
        nn.Linear(10, 10, bias=False),
        nn.ReLU(),
        nn.Linear(10, 3, bias=False),
    )
    x = torch.rand((1, 10))
    return model, x


def targeted_fgsm(model, x, target, eps):
    x_var = x.detach().clone().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x_var), torch.tensor([target]))
    grad = torch.autograd.grad(loss, x_var)[0]
    # A targeted step lowers the loss for the chosen class.
    return (x_var - eps * grad.sign()).clamp(0, 1).detach()


def targeted_iterative_attack(model, x, target=1, eps=0.35,
                              starts=16, steps=120):
    """Projected sign ascent on target logit minus the largest other logit."""
    generator = torch.Generator().manual_seed(1500)
    noise = 2 * torch.rand((starts, x.shape[1]), generator=generator) - 1
    candidates = (x + eps * noise).clamp(0, 1).detach()
    best_margin = float("-inf")
    best = None

    for _ in range(steps):
        candidates.requires_grad_(True)
        logits = model(candidates)
        other = torch.cat((logits[:, :target], logits[:, target + 1:]), dim=1)
        margins = logits[:, target] - other.max(dim=1).values
        index = margins.argmax().item()
        if margins[index].item() > best_margin:
            best_margin = margins[index].item()
            best = candidates[index:index + 1].detach().clone()

        grad = torch.autograd.grad(margins.sum(), candidates)[0]
        candidates = (candidates + eps * 0.025 * grad.sign()).detach()
        candidates = candidates.clamp(min=x - eps, max=x + eps).clamp(0, 1)

    return best


def report(label, model, x, x_adv):
    with torch.no_grad():
        logits = model(x_adv)[0]
        difference = x_adv - x
        print(f"{label}: class={logits.argmax().item()}, "
              f"logits={[round(v, 8) for v in logits.tolist()]}, "
              f"L_inf={difference.abs().max().item():.8f}, "
              f"L_2={torch.linalg.vector_norm(difference).item():.8f}")


if __name__ == "__main__":
    network, original = make_problem()
    report("Original", network, original, original)

    target_zero = targeted_fgsm(network, original, target=0, eps=0.5 - 1e-7)
    report("FGSM to class 0", network, original, target_zero)
    assert network(target_zero).argmax(dim=1).item() == 0
    assert (target_zero - original).abs().max().item() <= 0.5

    target_one_fgsm = targeted_fgsm(network, original, target=1,
                                    eps=0.5 - 1e-7)
    report("FGSM to class 1", network, original, target_one_fgsm)

    target_one_iterative = targeted_iterative_attack(network, original)
    report("Iterative to class 1", network, original, target_one_iterative)
    assert network(target_one_iterative).argmax(dim=1).item() == 1
    assert (target_one_iterative - original).abs().max().item() <= 0.5
