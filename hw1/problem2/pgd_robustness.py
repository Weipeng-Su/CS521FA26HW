"""Problem 2: evaluate the three checkpoints with L-infinity and L2 PGD."""

import argparse
import hashlib
import json
import pickle
import tarfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from model import PreActResNet18


EPS_INF = 8 / 255
EPS_L2 = 0.75
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"


def pgd_linf_untargeted(model: nn.Module, x: torch.Tensor,
                        labels: torch.Tensor, k: int, eps: float,
                        eps_step: float) -> torch.Tensor:
    """Untargeted PGD within the L-infinity budget."""
    model.eval()
    original = x.detach()
    adv = original.clone()
    for _ in range(k):
        adv.requires_grad_(True)
        grad = torch.autograd.grad(F.cross_entropy(model(adv), labels), adv)[0]
        adv = adv.detach() + eps_step * grad.sign()
        delta = (adv - original).clamp(-eps, eps)
        adv = (original + delta).clamp(0, 1).detach()
    return adv


def pgd_l2_untargeted(model: nn.Module, x: torch.Tensor,
                      labels: torch.Tensor, k: int, eps: float,
                      eps_step: float) -> torch.Tensor:
    """Untargeted PGD with an L2 projection for each image."""
    model.eval()
    original = x.detach()
    adv = original.clone()
    for _ in range(k):
        adv.requires_grad_(True)
        grad = torch.autograd.grad(F.cross_entropy(model(adv), labels), adv)[0]
        grad_norm = grad.flatten(1).norm(p=2, dim=1).clamp_min(1e-10)
        direction = grad / grad_norm[:, None, None, None]
        delta = adv.detach() + eps_step * direction - original
        delta_norm = delta.flatten(1).norm(p=2, dim=1).clamp_min(1e-10)
        scale = (eps / delta_norm).clamp(max=1)
        delta = delta * scale[:, None, None, None]
        adv = (original + delta).clamp(0, 1).detach()
    return adv


def load_cifar10_test(archive: Path) -> TensorDataset:
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != CIFAR10_MD5:
        raise ValueError(f"CIFAR-10 checksum mismatch: {digest}")
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.extractfile("cifar-10-batches-py/test_batch")
        if member is None:
            raise FileNotFoundError("CIFAR-10 test_batch is missing")
        record = pickle.load(member, encoding="bytes")
    images = torch.from_numpy(np.asarray(record[b"data"], dtype=np.uint8))
    images = images.reshape(-1, 3, 32, 32).float().div_(255)
    labels = torch.tensor(record[b"labels"], dtype=torch.long)
    return TensorDataset(images, labels)


def clean_count(model: nn.Module, loader: DataLoader, device: torch.device):
    correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += len(labels)
    return correct, total


def attack_counts(model: nn.Module, loader: DataLoader, device: torch.device,
                  steps: int):
    counts = {"clean": 0, "linf": 0, "l2": 0, "union": 0, "total": 0}
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        with torch.inference_mode():
            clean_prediction = model(images).argmax(1)
        with torch.enable_grad():
            linf = pgd_linf_untargeted(model, images, labels, steps,
                                        EPS_INF, EPS_INF / 4)
            l2 = pgd_l2_untargeted(model, images, labels, steps,
                                    EPS_L2, EPS_L2 / 4)
        with torch.inference_mode():
            linf_prediction = model(linf).argmax(1)
            l2_prediction = model(l2).argmax(1)
        correct_linf = linf_prediction == labels
        correct_l2 = l2_prediction == labels
        counts["clean"] += (clean_prediction == labels).sum().item()
        counts["linf"] += correct_linf.sum().item()
        counts["l2"] += correct_l2.sum().item()
        counts["union"] += (correct_linf & correct_l2).sum().item()
        counts["total"] += len(labels)
        if batch_index % 10 == 0 or batch_index == len(loader):
            print(f"  {batch_index}/{len(loader)} batches", flush=True)
    assert counts["union"] <= min(counts["linf"], counts["l2"])
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000,
                        help="Number of test images for adversarial evaluation")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(args.device)
    dataset = load_cifar10_test(args.data)
    if not 1 <= args.samples <= len(dataset):
        parser.error("--samples must be between 1 and 10000")
    if args.steps < 1:
        parser.error("--steps must be positive")

    rng = np.random.default_rng(42)
    indices = sorted(rng.choice(len(dataset), size=args.samples,
                                replace=False).tolist())
    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    attack_loader = DataLoader(Subset(dataset, indices),
                               batch_size=args.batch_size, shuffle=False)
    result = {
        "dataset": "CIFAR-10 official test set",
        "device": str(device),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "random_start": False,
        "step_size": "epsilon/4",
        "epsilon_linf": EPS_INF,
        "epsilon_l2": EPS_L2,
        "clean_test_size": len(dataset),
        "attack_sample_size": args.samples,
        "attack_sample_seed": 42,
        "models": {},
    }

    for name in ("Linf", "L2", "RAMP"):
        print(f"Evaluating {name} on {device}", flush=True)
        model = PreActResNet18().to(device)
        state = torch.load(args.models / f"pretr_{name}.pth",
                           map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        full_correct, full_total = clean_count(model, full_loader, device)
        counts = attack_counts(model, attack_loader, device, args.steps)
        result["models"][name] = {
            "clean_full": full_correct / full_total,
            "clean_attack_subset": counts["clean"] / counts["total"],
            "linf": counts["linf"] / counts["total"],
            "l2": counts["l2"] / counts["total"],
            "union": counts["union"] / counts["total"],
            "counts": counts,
        }
        print(json.dumps(result["models"][name], indent=2), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
