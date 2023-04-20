#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from torch._dynamo.test_case import run_tests, TestCase
import torch
from torch._dynamo.utils import counters
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA


class TestSplitCatFxPasses(TestCase):
    def test_split_normalization(self):
        def arg_only(x):
            return [torch.relu(s) for s in torch.split(x, 2, 1)]

        def kwarg1(x):
            return [torch.relu(s) for s in torch.split(x, 2, dim=1)]

        def kwarg2(x):
            return [
                torch.relu(s) for s in torch.split(x, split_size_or_sections=2, dim=1)
            ]

        def no_replace(x):
            return [torch.relu(s) for s in torch.split(x, [16, 16], dim=1)]

        def multi_split(x):
            return [torch.split(s, 2, 1) for s in torch.split(x, 2, 1)]

        def unequal_split(x):
            return [torch.relu(s) for s in torch.split(x, 3, 1)]

        torch._inductor.config.split_cat_pattern_matcher = True
        args = [
            torch.randn(2, 32, device="cuda"),
        ]
        for (fn, expected_split_norm_count) in [
            (arg_only, 1),
            (kwarg1, 1),
            (kwarg2, 1),
            (no_replace, 0),
            (multi_split, 17),
            (unequal_split, 1),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["split_cat_norm"],
                expected_split_norm_count,
            )
            counters.clear()


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA:
        run_tests()
