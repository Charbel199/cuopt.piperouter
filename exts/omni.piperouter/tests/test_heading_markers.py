"""set_marker_direction / get_world_axis round-trip against a real USD stage."""
import numpy as np
from pxr import Usd

from omni.piperouter import headings, scene_ops


def _stage_with_marker():
    stage = Usd.Stage.CreateInMemory()
    path = f"{scene_ops.MARKERS_SCOPE}/w0_start"
    scene_ops.spawn_marker(stage, path, (1.0, 2.0, 3.0), radius=0.05)
    return stage, path


def test_marker_direction_round_trip():
    stage, path = _stage_with_marker()
    for d in ((0, 0, 1), (1, 1, 0), (-1, 2, 0.5)):
        assert scene_ops.set_marker_direction(stage, path, d, show=True)
        ax = scene_ops.get_world_axis(stage, path)
        dn = np.asarray(d, float) / np.linalg.norm(d)
        assert ax is not None and np.allclose(ax, dn, atol=1e-5)
    # the arrow child exists while shown, is removed when hidden
    assert stage.GetPrimAtPath(f"{path}/dir").IsValid()
    scene_ops.set_marker_direction(stage, path, None, show=False)
    assert not stage.GetPrimAtPath(f"{path}/dir").IsValid()


def test_angles_through_marker_match_solver_vector():
    # panel flow: angles -> aim marker -> read world axis (what the solver receives)
    stage, path = _stage_with_marker()
    v = headings.angles_to_vector(35.0, 20.0, "Z")
    scene_ops.set_marker_direction(stage, path, v, show=True)
    ax = scene_ops.get_world_axis(stage, path)
    assert np.allclose(ax, v, atol=1e-5)


def test_missing_marker_is_safe():
    stage = Usd.Stage.CreateInMemory()
    assert scene_ops.set_marker_direction(stage, "/nope", (1, 0, 0)) is False
    assert scene_ops.get_world_axis(stage, "/nope") is None
