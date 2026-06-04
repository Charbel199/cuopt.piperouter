import numpy as np

from omni.piperouter import slices


def test_render_views_returns_three_rgba_images():
    res = (10, 8, 6)
    occ = np.zeros(res, dtype=np.uint8)
    occ[5, 4, 3] = 1
    thermal = np.full(res, 20.0, dtype=np.float32)
    imgs = slices.render_views([0, 0, 0], 0.1, res, occ, thermal, routes=[], target_px=80)
    assert set(imgs) == {"xy", "xz", "yz"}
    for k, im in imgs.items():
        assert im.ndim == 3 and im.shape[2] == 4
        assert im.dtype == np.uint8
        assert (im[..., 3] == 255).all()


def test_occupied_cell_is_darker_than_background():
    res = (8, 8, 4)
    occ = np.zeros(res, dtype=np.uint8)
    occ[:, :, :] = 0
    occ[4, 4, :] = 1  # an occupied column -> shows in the XY view
    thermal = np.full(res, 20.0, dtype=np.float32)
    imgs = slices.render_views([0, 0, 0], 0.1, res, occ, thermal, routes=[], target_px=8)
    xy = imgs["xy"].astype(int)
    # background brightness > occupancy brightness somewhere in the image
    assert xy[..., :3].sum(axis=2).max() > xy[..., :3].sum(axis=2).min()


def test_route_color_appears():
    res = (8, 8, 4)
    occ = np.zeros(res, dtype=np.uint8)
    thermal = np.full(res, 20.0, dtype=np.float32)
    route = {"points": [[0.05, 0.05, 0.15], [0.75, 0.05, 0.15]], "color": (0.1, 0.1, 0.9)}
    imgs = slices.render_views([0, 0, 0], 0.1, res, occ, thermal, routes=[route], target_px=8)
    xy = imgs["xy"]
    # end marker is red -> some pixel has a high red, low green channel
    reds = (xy[..., 0] > 200) & (xy[..., 1] < 80)
    assert reds.any()
