"""Tests for pairwise distances composed from sequence aligners."""

from sktime.alignment.lucky import AlignerLuckyDtw
from sktime.datasets import load_unit_test
from sktime.datatypes import convert_to
from sktime.dists_kernels.compose_from_align import DistFromAligner


def test_dist_from_aligner_preserves_asymmetric_ordered_distances():
    """DistFromAligner should not mirror distances from an asymmetric aligner."""
    X, _ = load_unit_test()
    X = X[0:3]
    X_list = convert_to(X, to_type="df-list", as_scitype="Panel")

    forward = AlignerLuckyDtw().fit([X_list[0], X_list[2]]).get_distance()
    reverse = AlignerLuckyDtw().fit([X_list[2], X_list[0]]).get_distance()

    assert forward != reverse

    dist = DistFromAligner(AlignerLuckyDtw())
    dist_mat = dist.transform(X)

    assert dist.get_tag("symmetric") is False
    assert dist_mat[0, 2] == forward
    assert dist_mat[2, 0] == reverse
