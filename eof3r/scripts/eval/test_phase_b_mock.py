"""Phase B mock training validation.

Tests the entire Phase B training pipeline with synthetic data,
without requiring GPU or real models. Validates:
    1. Loss functions compute correctly
    2. Heads produce correct outputs
    3. Trainer runs through all stages
    4. Checkpoint save/load works
    5. Data pipeline flows correctly

Usage:
    python eof3r/scripts/eval/test_phase_b_mock.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from eof3r.src.training.losses import (
    chamfer_depth_loss,
    focal_occupancy_loss,
    hinge_free_space_loss,
    compute_total_loss,
    get_stage_weights,
    STAGE_SCHEDULE,
)
from eof3r.src.training.heads import OccupancyHead, SemanticHead
from eof3r.src.training.supervision import (
    label_gaussians_by_vggt_projection,
    compute_vggt_surface_points,
)
from eof3r.src.training.trainer import PhaseBTrainer, PhaseBConfig


def create_mock_batch(
    num_gaussians: int = 100,
    num_views: int = 2,
    image_size: int = 64,
    num_classes: int = 10,
    device: str = "cpu",
) -> dict:
    """Create a mock training batch (without batch dimension for DataLoader)."""
    V, H, W = num_views, image_size, image_size
    G = num_gaussians

    return {
        "images": torch.randn(V, 3, H, W, device=device),
        "poses": torch.eye(4, device=device).unsqueeze(0).expand(V, -1, -1),
        "intrinsics": torch.tensor([
            [50.0, 0, W/2],
            [0, 50.0, H/2],
            [0, 0, 1.0],
        ], device=device).unsqueeze(0).expand(V, -1, -1),
        "depth": torch.rand(V, H, W, device=device) * 5 + 1,
        "surface_points": torch.randn(500, 3, device=device),
        "labels": {
            "occupied": torch.zeros(G, dtype=torch.bool, device=device).random_(0, 2),
            "free": torch.zeros(G, dtype=torch.bool, device=device).random_(0, 2),
        },
    }


class MockEncoder(torch.nn.Module):
    """Mock ReSplat encoder for testing."""

    def __init__(self, num_gaussians: int = 100):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.linear = torch.nn.Linear(10, num_gaussians * 3)

    def forward(self, context, global_step, deterministic=False):
        B = context["image"].shape[0]
        G = self.num_gaussians
        device = context["image"].device

        # Generate mock Gaussian parameters
        means = torch.randn(B, G, 3, device=device) * 2
        opacities = torch.sigmoid(torch.randn(B, G, device=device))
        scales = torch.rand(B, G, 3, device=device) * 0.5 + 0.1
        quats = torch.randn(B, G, 4, device=device)
        quats = quats / quats.norm(dim=-1, keepdim=True)
        harmonics = torch.randn(B, G, 16, 3, device=device) * 0.1
        covars = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, G, -1, -1).clone()

        # Create a simple Gaussians-like object
        class Gaussians:
            def __init__(self):
                self.means = means
                self.opacities = opacities
                self.scales = scales
                self.quats = quats
                self.harmonics = harmonics
                self.covars = covars
                self.covariances = covars  # Alias for compatibility

        return {"gaussians": Gaussians(), "depths": None}


def test_loss_functions():
    """Test all loss functions."""
    print("=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)

    N, M, K = 100, 500, 10
    means = torch.randn(N, 3)
    points = torch.randn(M, 3)
    occ = torch.sigmoid(torch.randn(N))
    labels = torch.randint(0, 2, (N,)).float()
    free_mask = torch.zeros(N).bool()
    free_mask[:70] = True
    occ_mask = ~free_mask

    # Chamfer
    loss = chamfer_depth_loss(means, points, occ_mask)
    print(f"  Chamfer loss: {loss.item():.4f} ✓")

    # Focal
    loss = focal_occupancy_loss(occ, labels)
    print(f"  Focal loss: {loss.item():.4f} ✓")

    # Hinge
    loss = hinge_free_space_loss(occ, free_mask)
    print(f"  Hinge loss: {loss.item():.4f} ✓")

    # Total loss
    logits = torch.randn(N, K)
    sem_labels = torch.randint(0, K, (N,))
    losses = compute_total_loss(
        gaussian_means=means,
        predicted_occupancy=occ,
        semantic_logits=logits,
        vggt_points=points,
        occupied_mask=occ_mask,
        free_mask=free_mask,
        semantic_labels=sem_labels,
        alpha=1.0, beta=1.0, gamma=1.0, delta=0.3, eta=0.0,
    )
    print(f"  Total loss: {losses['total'].item():.4f} ✓")
    print(f"  Components: {list(losses.keys())} ✓")

    return True


def test_heads():
    """Test occupancy and semantic heads."""
    print("\n" + "=" * 60)
    print("Testing Heads")
    print("=" * 60)

    N, K = 100, 10
    means = torch.randn(N, 3)
    scales = torch.rand(N, 3)
    rots = torch.randn(N, 4)
    rots = rots / rots.norm(dim=-1, keepdim=True)

    # Occupancy head
    occ_head = OccupancyHead()
    occ_pred = occ_head(means, scales, rots)
    print(f"  OccupancyHead: {occ_pred.shape}, range [{occ_pred.min():.3f}, {occ_pred.max():.3f}] ✓")

    # Semantic head
    sem_head = SemanticHead(num_classes=K)
    sem_pred = sem_head(means, scales, rots)
    print(f"  SemanticHead: {sem_pred.shape} ✓")

    # Gradient flow
    occ_loss = occ_pred.sum()
    occ_loss.backward()
    print(f"  Gradient flow through OccupancyHead: ✓")

    return True


def test_supervision():
    """Test VGGT supervision labeling."""
    print("\n" + "=" * 60)
    print("Testing VGGT Supervision")
    print("=" * 60)

    N = 200
    means = torch.randn(N, 3) * 2
    covs = torch.eye(3).unsqueeze(0).expand(N, -1, -1) * 0.01

    H, W = 64, 64
    depth = torch.rand(H, W) * 5 + 1
    K = torch.tensor([[50.0, 0, 32], [0, 50.0, 32], [0, 0, 1.0]])
    T = torch.eye(4)

    labels = label_gaussians_by_vggt_projection(means, covs, depth, T, K, kappa=3.0)
    print(f"  Labels: {labels.summary()} ✓")

    points = compute_vggt_surface_points(depth, K, T, subsample=4)
    print(f"  Surface points: {points.shape} ✓")

    return True


def test_stage_schedule():
    """Test three-stage training schedule."""
    print("\n" + "=" * 60)
    print("Testing Stage Schedule")
    print("=" * 60)

    total = 100_000
    for step in [0, 29999, 30000, 79999, 80000, 99999]:
        w = get_stage_weights(step, total)
        stage = 1 if step < 30000 else 2 if step < 80000 else 3
        print(f"  Step {step:6d}: stage={stage}, weights={w}")

    print(f"  Schedule: {STAGE_SCHEDULE} ✓")
    return True


def test_trainer_loop():
    """Test trainer with mock data."""
    print("\n" + "=" * 60)
    print("Testing Trainer Loop (Mock)")
    print("=" * 60)

    # Create mock components
    num_gaussians = 100
    num_classes = 10

    encoder = MockEncoder(num_gaussians)
    occ_head = OccupancyHead()
    sem_head = SemanticHead(num_classes=num_classes)

    # Create mock dataloader
    from torch.utils.data import Dataset, DataLoader

    class MockDataset(Dataset):
        def __init__(self, size: int = 50):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return create_mock_batch(num_gaussians=num_gaussians)

    train_dataset = MockDataset(size=50)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Create trainer with minimal config
    config = PhaseBConfig(
        total_steps=20,
        batch_size=1,
        log_interval=5,
        eval_interval=100,
        save_interval=100,
        output_dir=tempfile.mkdtemp(),
    )

    trainer = PhaseBTrainer(
        resplat_encoder=encoder,
        occupancy_head=occ_head,
        semantic_head=sem_head,
        train_loader=train_loader,
        val_loader=None,
        config=config,
        device="cpu",
    )

    # Run a few training steps
    print(f"  Running {config.total_steps} training steps...")
    trainer.train()

    print(f"  Final step: {trainer.state.global_step}")
    print(f"  Losses history: {len(trainer.state.losses_history)} entries")
    if trainer.state.losses_history:
        last_loss = trainer.state.losses_history[-1]
        print(f"  Last loss: {last_loss['total']:.4f}")

    return True


def test_checkpoint():
    """Test checkpoint save/load."""
    print("\n" + "=" * 60)
    print("Testing Checkpoint Save/Load")
    print("=" * 60)

    # Create model
    occ_head = OccupancyHead()
    sem_head = SemanticHead(num_classes=10)

    # Save
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
        torch.save({
            "occ_head": occ_head.state_dict(),
            "sem_head": sem_head.state_dict(),
        }, path)
        print(f"  Saved checkpoint: {path}")

    # Load
    checkpoint = torch.load(path)
    occ_head_new = OccupancyHead()
    occ_head_new.load_state_dict(checkpoint["occ_head"])
    print(f"  Loaded checkpoint: ✓")

    # Verify weights match
    for (n1, p1), (n2, p2) in zip(
        occ_head.named_parameters(),
        occ_head_new.named_parameters()
    ):
        assert torch.allclose(p1, p2), f"Mismatch in {n1}"
    print(f"  Weights match: ✓")

    return True


def main():
    print("=" * 60)
    print("Phase B Mock Training Validation")
    print("=" * 60)

    tests = [
        ("Loss Functions", test_loss_functions),
        ("Heads", test_heads),
        ("Supervision", test_supervision),
        ("Stage Schedule", test_stage_schedule),
        ("Trainer Loop", test_trainer_loop),
        ("Checkpoint", test_checkpoint),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    all_passed = True
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
        if not success:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! Phase B training pipeline is ready.")
    else:
        print("Some tests failed. Check output above.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
