"""Kiểm tra tự viết cho inference.py (RUBRIC mục H). Chạy trong thư mục code/, không cần GPU:
    python -m unittest test_inference -v
"""
import unittest

import numpy as np
import torch
from torch import nn

import inference
import model as model_lib

K = 9


def softmax(z):
    return torch.as_tensor(z, dtype=torch.float64).softmax(-1).numpy()


class TestViews(unittest.TestCase):
    def setUp(self):
        self.x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).view(2, 3, 8, 8)

    def test_hflip(self):
        f = inference.view_hflip(self.x)
        torch.testing.assert_close(f[..., 0], self.x[..., -1])
        torch.testing.assert_close(inference.view_hflip(f), self.x)

    def test_multicrop(self):
        crops = inference.views_multicrop(self.x, 6)
        self.assertEqual(len(crops), 5)
        self.assertTrue(all(c.shape == (2, 3, 6, 6) for c in crops))
        torch.testing.assert_close(crops[0], self.x[..., :6, :6])      # góc trên trái
        torch.testing.assert_close(crops[3], self.x[..., 2:, 2:])      # góc dưới phải
        torch.testing.assert_close(crops[4], self.x[..., 1:7, 1:7])    # giữa
        self.assertEqual(len(inference.views_multicrop(self.x, 6, flip=True)), 10)

    def test_multiscale(self):
        out = inference.views_multiscale(self.x, [8, 12, 6])
        self.assertEqual([o.shape[-1] for o in out], [8, 12, 6])
        self.assertIs(out[0], self.x)


class TestAggregation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.views = [rng.normal(size=(50, K)) * 3 for _ in range(4)]

    def test_prob_and_logit_space(self):
        prob = inference.aggregate_views(self.views, "prob")
        logit = inference.aggregate_views(self.views, "logit")
        np.testing.assert_allclose(prob.sum(1), 1.0, atol=1e-9)
        np.testing.assert_allclose(logit.sum(1), 1.0, atol=1e-9)
        np.testing.assert_allclose(prob, np.mean([softmax(v) for v in self.views], axis=0), atol=1e-9)
        np.testing.assert_allclose(logit, softmax(np.mean(self.views, axis=0)), atol=1e-9)
        self.assertGreater(np.abs(prob - logit).max(), 1e-3)     # hai cách gộp thật sự khác nhau

    def test_single_view_is_softmax(self):
        np.testing.assert_allclose(inference.aggregate_views(self.views[:1], "prob"), softmax(self.views[0]))

    def test_ensemble(self):
        probs = [softmax(v) for v in self.views]
        np.testing.assert_allclose(inference.ensemble_probs(probs), np.mean(probs, axis=0))
        with self.assertRaises(ValueError):
            inference.ensemble_probs([probs[0], probs[1][:10]])


class TestTemperature(unittest.TestCase):
    def test_recovers_known_temperature(self):
        # nhãn lấy mẫu từ softmax(z), rồi đưa cho hàm logit bị nhân 3 (quá tự tin): T tối ưu phải gần 3
        g = torch.Generator().manual_seed(0)
        z = torch.randn(20000, K, generator=g, dtype=torch.float64) * 2
        y = torch.multinomial(z.softmax(1), 1, generator=g).squeeze(1)
        T = inference.fit_temperature((3 * z).numpy(), y.numpy())
        self.assertAlmostEqual(T, 3.0, delta=0.1)

    def test_temperature_keeps_argmax_and_lowers_nll(self):
        g = torch.Generator().manual_seed(1)
        z = torch.randn(5000, K, generator=g, dtype=torch.float64) * 2
        y = torch.multinomial(z.softmax(1), 1, generator=g).squeeze(1).numpy()
        logits = (4 * z).numpy()
        T = inference.fit_temperature(logits, y)
        before, after = inference.apply_temperature(logits, 1.0), inference.apply_temperature(logits, T)
        np.testing.assert_array_equal(before.argmax(1), after.argmax(1))
        nll = lambda p: -np.log(p[np.arange(len(y)), y]).mean()
        self.assertLess(nll(after), nll(before))


class TestFuseBN(unittest.TestCase):
    def _randomize_bn(self, net):
        g = torch.Generator().manual_seed(0)
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.copy_(torch.randn(m.num_features, generator=g) * 0.1)
                m.running_var.copy_(torch.rand(m.num_features, generator=g) + 0.5)
                m.weight.data.copy_(torch.rand(m.num_features, generator=g) + 0.5)
                m.bias.data.copy_(torch.randn(m.num_features, generator=g) * 0.1)

    def _check(self, name, tol):
        torch.manual_seed(0)
        net = model_lib.build_model(name, pretrained=False).double().eval()
        self._randomize_bn(net)
        fused = inference.fuse_conv_bn(net)
        self.assertEqual(sum(isinstance(m, nn.BatchNorm2d) for m in fused.modules()), 0, "còn BN chưa gộp")
        self.assertGreater(sum(isinstance(m, nn.BatchNorm2d) for m in net.modules()), 0, "model gốc bị sửa")
        self.assertLess(inference.fuse_error(net, fused, img_size=64), tol)

    def test_resnet50(self):
        self._check("resnet50", 1e-8)

    def test_mobilenetv3_keeps_activation(self):
        self._check("mobilenetv3", 1e-8)

    def test_efficientnet_b0(self):
        self._check("efficientnet_b0", 1e-8)

    def test_no_bn_model_is_unchanged(self):
        net = model_lib.build_model("convnext_tiny", pretrained=False).eval()
        fused = inference.fuse_conv_bn(net)
        self.assertEqual(inference.fuse_error(net, fused, img_size=64), 0.0)


class TestSoup(unittest.TestCase):
    def test_uniform_soup(self):
        a, b = nn.BatchNorm2d(3), nn.BatchNorm2d(3)
        b.weight.data.fill_(3.0)
        soup = inference.uniform_soup([a.state_dict(), b.state_dict()])
        torch.testing.assert_close(soup["weight"], torch.full((3,), 2.0))
        self.assertEqual(soup["num_batches_tracked"].dtype, torch.long)


class TestPredict(unittest.TestCase):
    def test_predict_views_order_and_flip(self):
        torch.manual_seed(0)
        net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, K))
        xs = torch.randn(10, 3, 4, 4)
        loader = [(xs[i:i + 4], torch.arange(i, min(i + 4, 10)), [f"{j}.jpg" for j in range(i, min(i + 4, 10))])
                  for i in range(0, 10, 4)]
        dev = torch.device("cpu")
        names, y, (plain, flipped) = inference.predict_views(net, loader, dev,
                                                             [inference.view_identity, inference.view_hflip])
        self.assertEqual(names, [f"{j}.jpg" for j in range(10)])
        np.testing.assert_array_equal(y, np.arange(10))
        with torch.no_grad():
            np.testing.assert_allclose(plain, net(xs).numpy(), atol=1e-6)
            np.testing.assert_allclose(flipped, net(torch.flip(xs, [-1])).numpy(), atol=1e-6)
        np.testing.assert_allclose(inference.predict_logits(net, loader, dev)[2], plain)


if __name__ == "__main__":
    unittest.main()
