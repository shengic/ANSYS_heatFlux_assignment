from __future__ import annotations

import numpy as np
import pytest

from heatflux.model.ansys_node_store import AnsysNodeStore


def test_get_xyz_and_batch() -> None:
    ids = np.array([10, 20, 30], dtype=np.int32)
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    store = AnsysNodeStore(node_ids=ids, xyz=xyz)
    assert np.allclose(store.get_xyz(20), np.array([4.0, 5.0, 6.0]))
    assert np.allclose(
        store.get_xyz_batch([30, 10]),
        np.array([[7.0, 8.0, 9.0], [1.0, 2.0, 3.0]]),
    )


def test_unknown_node_id_raises() -> None:
    ids = np.array([1], dtype=np.int32)
    xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    store = AnsysNodeStore(node_ids=ids, xyz=xyz)
    with pytest.raises(KeyError):
        _ = store.get_xyz(2)
