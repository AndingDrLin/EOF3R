"""Tests for Phase B training module.

Verifies:
    1. Loss functions compute correctly and are differentiable
    2. Occupancy/Semantic heads produce correct output shapes
    3. VGGT supervision labeling works
    4. Three-stage schedule produces correct weights
    5. Total loss composition works end-to-end

Usage:
    python eof3r/tests/test_training.py
    pytest eof3r/tests/test_training.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from eof3r.src.training.losses import (
    chamfer_depth_loss,
    focal_occupancy_loss,
    hinge_free_space_loss,
    semantic_cross_entropy_loss,
    color_reconstruction_loss,
    compute_total_loss,
    get_stage_weights,
    STAGE_SCHEDULE,
)
from eof3r.src.training.heads import OccupancyHead, SemanticHead, ConfidenceHead
from eof3r.src.training.supervision import (
    label_gaussians_by_vggt_projection,
    compute_vggt_surface_points,
    merge_multi_view_labels,
    GaussianLabels,
)


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

class TestChamferDepthLoss:
    def test_basic(self):
        means = torch.randn(50, 3)
        points = torch.randn(200, 3)
        loss = chamfer_depth_loss(means, points)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0

    def test_with_mask(self):
        means = torch.randn(50, 3)
        points = torch.randn(200, 3)
        mask = torch.zeros(50).bool()
        mask[30:] = True
        loss = chamfer_depth_loss(means, points, mask)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_gradient(self):
        means = torch.randn(50, 3, requires_grad=True)
        points = torch.randn(200, 3)
        loss = chamfer_depth_loss(means, points)
        loss.backward()
        assert means.grad is not None
        assert means.grad.shape == means.shape

    def test_empty_mask(self):
        means = torch.randn(50, 3)
        points = torch.randn(200, 3)
        mask = torch.zeros(50).bool()
        loss = chamfer_depth_loss(means, points, mask)
        assert loss.item() > 0  # forward term still applies

    def test_close_points_low_loss(self):
        # If means and points are the same, loss should be ~0
        points = torch.randn(50, 3)
        loss = chamfer_depth_loss(points.clone(), points)
        assert loss.item() < 0.01


class TestFocalOccupancyLoss:
    def test_basic(self):
        pred = torch.sigmoid(torch.randn(100))
        labels = torch.randint(0, 2, (100,)).float()
        loss = focal_occupancy_loss(pred, labels)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_gradient(self):
        # Use raw logits and apply sigmoid inside to get leaf tensor gradient
        logits = torch.randn(100, requires_grad=True)
        pred = torch.sigmoid(logits)
        labels = torch.randint(0, 2, (100,)).float()
        loss = focal_occupancy_loss(pred, labels)
        loss.backward()
        assert logits.grad is not None

    def test_perfect_prediction(self):
        # If prediction matches label perfectly, loss should be low
        pred = torch.ones(100) * 0.99
        labels = torch.ones(100)
        loss = focal_occupancy_loss(pred, labels)
        assert loss.item() < 0.1

    def test_gamma_effect(self):
        pred = torch.sigmoid(torch.randn(100))
        labels = torch.randint(0, 2, (100,)).float()
        loss_g0 = focal_occupancy_loss(pred, labels, gamma=0.0)
        loss_g2 = focal_occupancy_loss(pred, labels, gamma=2.0)
        # Both should be positive
        assert loss_g0.item() > 0
        assert loss_g2.item() > 0


class TestHingeFreeSpaceLoss:
    def test_basic(self):
        pred = torch.sigmoid(torch.randn(100))
        mask = torch.zeros(100).bool()
        mask[:70] = True
        loss = hinge_free_space_loss(pred, mask)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_below_epsilon_no_loss(self):
        pred = torch.ones(100) * 0.01  # well below epsilon=0.05
        mask = torch.ones(100).bool()
        loss = hinge_free_space_loss(pred, mask)
        assert loss.item() < 1e-6

    def test_above_epsilon_has_loss(self):
        pred = torch.ones(100) * 0.5  # well above epsilon
        mask = torch.ones(100).bool()
        loss = hinge_free_space_loss(pred, mask)
        assert loss.item() > 0

    def test_empty_mask(self):
        pred = torch.ones(100) * 0.5
        mask = torch.zeros(100).bool()
        loss = hinge_free_space_loss(pred, mask)
        assert loss.item() == 0.0


class TestSemanticCrossEntropyLoss:
    def test_basic(self):
        logits = torch.randn(100, 10)
        labels = torch.randint(0, 10, (100,))
        mask = torch.ones(100).bool()
        loss = semantic_cross_entropy_loss(logits, labels, mask)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_empty_mask(self):
        logits = torch.randn(100, 10)
        labels = torch.randint(0, 10, (100,))
        mask = torch.zeros(100).bool()
        loss = semantic_cross_entropy_loss(logits, labels, mask)
        assert loss.item() == 0.0


class TestColorReconstructionLoss:
    def test_basic(self):
        rendered = torch.rand(1, 3, 64, 64)
        gt = torch.rand(1, 3, 64, 64)
        loss = color_reconstruction_loss(rendered, gt)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_identical_images(self):
        img = torch.rand(1, 3, 64, 64)
        loss = color_reconstruction_loss(img, img)
        assert loss.item() < 0.01


class TestComputeTotalLoss:
    def test_basic(self):
        N, M, K = 50, 200, 10
        means = torch.randn(N, 3)
        occ = torch.sigmoid(torch.randn(N))
        logits = torch.randn(N, K)
        points = torch.randn(M, 3)
        occ_mask = torch.zeros(N).bool()
        occ_mask[30:] = True
        free_mask = torch.zeros(N).bool()
        free_mask[:30] = True
        sem_labels = torch.randint(0, K, (N,))

        losses = compute_total_loss(
            gaussian_means=means,
            predicted_occupancy=occ,
            semantic_logits=logits,
            vggt_points=points,
            occupied_mask=occ_mask,
            free_mask=free_mask,
            semantic_labels=sem_labels,
            alpha=1.0, beta=1.0, gamma=1.0, delta=0.3, eta=0.1,
        )

        assert "total" in losses
        assert "depth" in losses
        assert "occupancy" in losses
        assert "free_space" in losses
        assert "semantic" in losses
        assert losses["total"].ndim == 0


# ---------------------------------------------------------------------------
# Head tests
# ---------------------------------------------------------------------------

class TestOccupancyHead:
    def test_output_shape(self):
        head = OccupancyHead()
        means = torch.randn(100, 3)
        scales = torch.rand(100, 3)
        rots = torch.randn(100, 4)
        rots = rots / rots.norm(dim=-1, keepdim=True)

        out = head(means, scales, rots)
        assert out.shape == (100,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_gradient(self):
        head = OccupancyHead()
        means = torch.randn(50, 3, requires_grad=True)
        scales = torch.rand(50, 3)
        rots = torch.randn(50, 4)
        rots = rots / rots.norm(dim=-1, keepdim=True)

        out = head(means, scales, rots)
        out.sum().backward()
        assert means.grad is not None

    def test_custom_hidden_dims(self):
        head = OccupancyHead(hidden_dims=[128, 64, 32])
        means = torch.randn(50, 3)
        scales = torch.rand(50, 3)
        rots = torch.randn(50, 4)
        out = head(means, scales, rots)
        assert out.shape == (50,)


class TestSemanticHead:
    def test_output_shape(self):
        head = SemanticHead(num_classes=10)
        means = torch.randn(100, 3)
        scales = torch.rand(100, 3)
        rots = torch.randn(100, 4)

        out = head(means, scales, rots)
        assert out.shape == (100, 10)


class TestConfidenceHead:
    def test_output_shape(self):
        head = ConfidenceHead()
        means = torch.randn(100, 3)
        scales = torch.rand(100, 3)
        rots = torch.randn(100, 4)

        out = head(means, scales, rots)
        assert out.shape == (100,)
        assert (out >= 0).all() and (out <= 1).all()


# ---------------------------------------------------------------------------
# Supervision tests
# ---------------------------------------------------------------------------

class TestGaussianLabels:
    def test_summary(self):
        labels = GaussianLabels(
            occupied=torch.tensor([True, False, False]),
            free=torch.tensor([False, True, False]),
            unknown=torch.tensor([False, False, True]),
            depth_error=torch.zeros(3),
            threshold=torch.ones(3),
        )
        assert labels.num_occupied == 1
        assert labels.num_free == 1
        assert labels.num_unknown == 1
        assert labels.num_labeled == 2
        assert "total=3" in labels.summary()


class TestLabelGaussiansByVGGTProjection:
    def test_basic(self):
        N = 100
        means = torch.randn(N, 3)
        covs = torch.eye(3).unsqueeze(0).expand(N, -1, -1) * 0.01
        depth = torch.rand(64, 64) * 5 + 1
        K = torch.tensor([[50.0, 0, 32], [0, 50.0, 32], [0, 0, 1.0]])
        T = torch.eye(4)

        labels = label_gaussians_by_vggt_projection(means, covs, depth, T, K)
        assert isinstance(labels, GaussianLabels)
        assert labels.occupied.shape == (N,)

    def test_all_unknown_when_out_of_bounds(self):
        # Place Gaussians far from camera
        means = torch.randn(100, 3) * 100
        covs = torch.eye(3).unsqueeze(0).expand(100, -1, -1) * 0.01
        depth = torch.rand(64, 64) * 5 + 1
        K = torch.tensor([[50.0, 0, 32], [0, 50.0, 32], [0, 0, 1.0]])
        T = torch.eye(4)

        labels = label_gaussians_by_vggt_projection(means, covs, depth, T, K)
        # Most should be unknown (out of bounds)
        assert labels.num_unknown > 50


class TestComputeVGGTSurfacePoints:
    def test_basic(self):
        depth = torch.rand(64, 64) * 5 + 1
        K = torch.tensor([[50.0, 0, 32], [0, 50.0, 32], [0, 0, 1.0]])
        T = torch.eye(4)

        points = compute_vggt_surface_points(depth, K, T, subsample=4)
        assert points.ndim == 2
        assert points.shape[1] == 3

    def test_subsample(self):
        depth = torch.rand(64, 64) * 5 + 1
        K = torch.tensor([[50.0, 0, 32], [0, 50.0, 32], [0, 0, 1.0]])
        T = torch.eye(4)

        points_full = compute_vggt_surface_points(depth, K, T, subsample=1)
        points_sub = compute_vggt_surface_points(depth, K, T, subsample=4)
        assert points_sub.shape[0] < points_full.shape[0]


# ---------------------------------------------------------------------------
# Schedule tests
# ---------------------------------------------------------------------------

class TestStageSchedule:
    def test_stages(self):
        total = 100_000
        for step in [0, 29999]:
            w = get_stage_weights(step, total)
            assert w["alpha"] == 1.0  # Stage 1

        for step in [30000, 79999]:
            w = get_stage_weights(step, total)
            assert w["beta"] == 1.0  # Stage 2

        for step in [80000, 99999]:
            w = get_stage_weights(step, total)
            assert w["gamma"] == 1.0  # Stage 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
