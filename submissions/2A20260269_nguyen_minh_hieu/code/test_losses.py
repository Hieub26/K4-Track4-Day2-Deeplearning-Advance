"""Kiểm tra tự viết cho losses.py (RUBRIC mục C và H). Chạy trong thư mục code/:
    python -m unittest test_losses -v
"""
import unittest

import torch
import torch.nn.functional as F

import losses

K = 9


class TestCriteria(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.logits = torch.randn(64, K, generator=g) * 3
        self.y = torch.randint(0, K, (64,), generator=g)
        self.ce = F.cross_entropy(self.logits, self.y)

    def test_focal_gamma_zero_is_ce(self):
        self.assertLess(abs(losses.FocalLoss(gamma=0.0)(self.logits, self.y) - self.ce), 1e-6)

    def test_focal_downweights_easy_examples(self):
        self.assertLess(losses.FocalLoss(gamma=2.0)(self.logits, self.y), self.ce)
        # mẫu đã đoán đúng chắc chắn gần như không đóng góp
        easy = torch.full((1, K), -10.0)
        easy[0, 3] = 10.0
        self.assertLess(losses.FocalLoss(gamma=2.0)(easy, torch.tensor([3])), 1e-12)

    def test_focal_alpha_matches_weighted_sum(self):
        alpha = torch.linspace(0.5, 2.0, K)
        got = losses.FocalLoss(gamma=0.0, alpha=alpha)(self.logits, self.y)
        want = (alpha[self.y] * F.cross_entropy(self.logits, self.y, reduction="none")).mean()
        self.assertLess(abs(got - want), 1e-6)

    def test_label_smoothing_zero_is_ce(self):
        self.assertLess(abs(losses.LabelSmoothingCE(0.0)(self.logits, self.y) - self.ce), 1e-6)

    def test_label_smoothing_matches_torch(self):
        want = F.cross_entropy(self.logits, self.y, label_smoothing=0.1)
        self.assertLess(abs(losses.LabelSmoothingCE(0.1)(self.logits, self.y) - want), 1e-6)

    def test_build_criterion(self):
        self.assertLess(abs(losses.build_criterion("ce")(self.logits, self.y) - self.ce), 1e-6)
        self.assertIsInstance(losses.build_criterion("ls", smoothing=0.1), losses.LabelSmoothingCE)
        self.assertEqual(losses.build_criterion("focal", gamma=1.5).gamma, 1.5)
        w = losses.class_weights([675, 637, 618, 613, 637, 605, 644, 609, 5463])
        want = F.cross_entropy(self.logits, self.y, weight=w)
        self.assertLess(abs(losses.build_criterion("ce_weighted", weight=w)(self.logits, self.y) - want), 1e-6)
        with self.assertRaises(ValueError):
            losses.build_criterion("ce_weighted")
        with self.assertRaises(ValueError):
            losses.build_criterion("khong_co")


class TestClassWeights(unittest.TestCase):
    counts = [675, 637, 618, 613, 637, 605, 644, 609, 5463]   # train fold 0

    def test_inverse_frequency(self):
        w = losses.class_weights(self.counts)
        self.assertAlmostEqual(float(w.sum()), K, places=4)
        self.assertAlmostEqual(float(w[0] / w[8]), 5463 / 675, places=3)

    def test_class_balanced(self):
        w = losses.class_weights(self.counts, beta=0.999)
        self.assertAlmostEqual(float(w.sum()), K, places=4)
        # lớp hiếm vẫn nặng hơn, nhưng chênh lệch nhỏ hơn 1/n
        self.assertGreater(float(w[0]), float(w[8]))
        self.assertLess(float(w[0] / w[8]), 5463 / 675)


class TestMix(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.arange(8, dtype=torch.float32).view(8, 1, 1, 1).expand(8, 3, 32, 32).contiguous()
        self.y = torch.arange(8)

    def test_mixup(self):
        x_mix, (y_a, y_b, lam) = losses.mix_batch(self.x, self.y, 1.0, "mixup")
        self.assertTrue(0.0 <= lam <= 1.0)
        torch.testing.assert_close(x_mix[:, 0, 0, 0], lam * y_a.float() + (1 - lam) * y_b.float())

    def test_cutmix_lambda_is_true_area(self):
        for _ in range(50):
            x_mix, (y_a, y_b, lam) = losses.mix_batch(self.x, self.y, 1.0, "cutmix")
            # mỗi ảnh mang giá trị bằng nhãn của nó, nên điểm ảnh nào bằng y_a là điểm còn giữ nguyên
            moved = y_a != y_b
            kept = (x_mix[moved, 0] == y_a[moved].float().view(-1, 1, 1)).float().mean(dim=(1, 2))
            torch.testing.assert_close(kept, torch.full_like(kept, lam), atol=1e-6, rtol=0)
            pasted = (x_mix[moved, 0] == y_b[moved].float().view(-1, 1, 1)).float().mean(dim=(1, 2))
            torch.testing.assert_close(pasted, torch.full_like(pasted, 1 - lam), atol=1e-6, rtol=0)

    def test_input_not_modified(self):
        before = self.x.clone()
        losses.mix_batch(self.x, self.y, 1.0, "cutmix")
        losses.mix_batch(self.x, self.y, 1.0, "mixup")
        torch.testing.assert_close(self.x, before)

    def test_mixed_loss_mixes_labels(self):
        logits = torch.randn(8, K)
        y_b = self.y.flip(0)
        ce = torch.nn.CrossEntropyLoss()
        got = losses.mixed_loss(ce, logits, (self.y, y_b, 0.3))
        self.assertLess(abs(got - (0.3 * ce(logits, self.y) + 0.7 * ce(logits, y_b))), 1e-6)
        # với CE, tổng có trọng số của hai CE bằng CE với nhãn mềm
        soft = 0.3 * F.one_hot(self.y, K) + 0.7 * F.one_hot(y_b, K)
        self.assertLess(abs(got - F.cross_entropy(logits, soft)), 1e-6)


if __name__ == "__main__":
    unittest.main()
