"""Operator tests (issue B23): plane, sine, constant and known-RMS surfaces, resolution check, NaN rule."""

import math

import numpy as np
import pytest

from gsop import operators as O

TILE = 6000.0
SIDE = 18000.0  # 3 x 3 tiles; the central tile is fully valid even though the array border has no slope

# --- named tolerances, each with its reason -----------------------------------------------------------------------
TOL_EXACT = 1e-9  # exact arithmetic on exactly representable surfaces (constant, plane)
TOL_PLANE_DEG = 1e-6  # a plane's slope is exact up to float rounding in arctan
TOL_PLANE_ROUGH_M = 1e-3  # 1 mm: float rounding of the closed-form roughness on a plane whose heights reach kilometres
TOL_SINE_ROUGH_A_CONTINUOUS = 0.02  # 2%: 60 point samples per tile side against the continuous analytic RMS
TOL_SINE_DISCRETE = 1e-9  # exact discrete expectation (point sampling, block means, line fit); only rounding remains
TOL_SINE_RELIEF = 1e-9  # the 500 m cell centres hit the sine's peaks and troughs exactly (6 cells per wavelength)
TOL_RESOLUTION = 0.01  # 1%: the same smooth surface sampled at 50 m and 250 m, both block-averaged to 500 m
TOL_HORN = 1e-9  # Horn is exact on a plane


def grid(cell_m, side_m=SIDE):
    n = int(round(side_m / cell_m))
    x = (np.arange(n) + 0.5) * cell_m
    return np.meshgrid(x, x)  # X east, Y south-to-north irrelevant for these surfaces


def centre_tile(a):
    """The fully valid central tile of a 3 x 3 result."""
    assert a.shape == (3, 3)
    return a[1, 1]


# --- constant ------------------------------------------------------------------------------------------------------
def test_constant_surface_gives_zero_everywhere():
    dem = np.full((180, 180), 321.0)  # 100 m cells, 18 km
    out = O.tile_features(dem, 100.0)
    for key in ("slope_med", "slope_p90", "relief", "rough_a", "rough_b"):
        assert out[key].shape == (3, 3)
        assert centre_tile(out[key]) == pytest.approx(0.0, abs=TOL_EXACT), key


# --- tilted plane ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("angle_deg", [1.0, 8.0, 20.0, 45.0])
def test_tilted_plane_slope_equals_the_angle_and_plane_removal_gives_zero_roughness(angle_deg):
    X, Y = grid(100.0)
    # tilt towards an arbitrary horizontal direction: slope angle is the angle of the gradient magnitude
    gx, gy = math.tan(math.radians(angle_deg)) * math.cos(0.7), math.tan(math.radians(angle_deg)) * math.sin(0.7)
    dem = gx * X + gy * Y
    out = O.tile_features(dem, 100.0)
    assert centre_tile(out["slope_med"]) == pytest.approx(angle_deg, abs=TOL_PLANE_DEG)
    assert centre_tile(out["slope_p90"]) == pytest.approx(angle_deg, abs=TOL_PLANE_DEG)
    assert centre_tile(out["rough_a"]) == pytest.approx(0.0, abs=TOL_PLANE_ROUGH_M)  # plane removal leaves nothing
    assert centre_tile(out["rough_b"]) == pytest.approx(0.0, abs=TOL_PLANE_ROUGH_M)
    # relief of a tilted plane at the 500 m baseline: the 12 cell means span 11 cells along the gradient direction
    span = abs(gx) * 11 * 500.0 + abs(gy) * 11 * 500.0
    assert centre_tile(out["relief"]) == pytest.approx(span, rel=1e-9)


def test_horn_is_exact_on_a_plane_and_sees_both_axes():
    X, Y = grid(500.0, 6000.0)
    slope = O.horn_slope_deg(0.1 * X + 0.2 * Y, 500.0)
    assert np.nanmax(np.abs(slope - math.degrees(math.atan(math.hypot(0.1, 0.2))))[1:-1, 1:-1]) < TOL_HORN
    assert np.isnan(slope[0, :]).all() and np.isnan(slope[:, -1]).all()  # no neighbourhood on the border


# --- sine with known amplitude and wavelength -------------------------------------------------------------------------
WAVELENGTH = 3000.0  # two wavelengths per 6 km tile
AMP = 40.0


def sine_dem():
    X, _ = grid(100.0)
    return AMP * np.sin(2 * np.pi * X / WAVELENGTH)


def attenuation(native_m, block_cells):
    """Amplitude factor of the mean of ``block_cells`` point samples spaced ``native_m`` of a sinusoid of WAVELENGTH
    (a Dirichlet kernel, the discrete version of sinc)."""
    u = math.pi * native_m / WAVELENGTH
    return 1.0 if block_cells == 1 else math.sin(block_cells * u) / (block_cells * math.sin(u))


def analytic_rough(amplitude, periods_per_tile=2):
    """Continuous RMS after plane removal of A sin(kx) over exactly m periods: A * sqrt(1/2 - 12 / (k L)^2), k L = 2 pi m."""
    return amplitude * math.sqrt(0.5 - 12.0 / (2 * math.pi * periods_per_tile) ** 2)


def discrete_rough(block_cells, native_m=100.0):
    """Independent expectation by line fit (np.polyfit, not the operator's closed form) of the block-mean profile of one tile."""
    step = block_cells * native_m
    n = int(round(TILE / step))
    x = (np.arange(n) + 0.5) * step
    z = AMP * attenuation(native_m, block_cells) * np.sin(2 * np.pi * x / WAVELENGTH)
    resid = z - np.polyval(np.polyfit(x, z, 1), x)
    return float(np.sqrt(np.mean(resid**2)))


def test_sine_relief_matches_the_analytic_value_at_the_500_m_baseline():
    out = O.tile_features(sine_dem(), 100.0)
    assert centre_tile(out["relief"]) == pytest.approx(2 * AMP * attenuation(100.0, 5), abs=TOL_SINE_RELIEF * 2 * AMP)


def test_sine_roughness_matches_the_analytic_rms_after_plane_removal():
    out = O.tile_features(sine_dem(), 100.0)
    # exact discrete expectation, both baselines
    assert centre_tile(out["rough_a"]) == pytest.approx(discrete_rough(1), rel=TOL_SINE_DISCRETE)
    assert centre_tile(out["rough_b"]) == pytest.approx(discrete_rough(10), rel=TOL_SINE_DISCRETE)
    # and the continuous formula for the fine baseline
    assert centre_tile(out["rough_a"]) == pytest.approx(analytic_rough(AMP), rel=TOL_SINE_ROUGH_A_CONTINUOUS)
    assert centre_tile(out["rough_b"]) < centre_tile(out["rough_a"])  # a coarser baseline smooths a short wavelength


def test_plane_removal_matters_for_the_sine():
    # without plane removal the RMS would be A/sqrt(2); the operator must be clearly below that
    out = O.tile_features(sine_dem(), 100.0)
    assert centre_tile(out["rough_a"]) < 0.95 * AMP / math.sqrt(2)


def test_closed_form_roughness_equals_least_squares_on_random_full_windows():
    rng = np.random.default_rng(3)
    win = rng.normal(size=(5, 12, 12)) * 7 + np.add.outer(np.arange(12) * 2.0, np.arange(12) * -1.5)
    closed = O._closed_form_rms(win)
    for i in range(5):
        assert closed[i] == pytest.approx(O.plane_rms(win[i]), rel=1e-9)


# --- resolution check -----------------------------------------------------------------------------------------------------
def smooth_surface(cell_m, side_m=24000.0):
    X, Y = grid(cell_m, side_m)
    return 120 * np.sin(2 * np.pi * X / 20000.0) + 80 * np.sin(2 * np.pi * Y / 15000.0 + 0.4)


def test_same_surface_at_two_native_resolutions_gives_the_same_slope_after_resampling_to_500_m():
    fine = O.tile_features(smooth_surface(50.0), 50.0, rough_baselines_m=(500.0, 1000.0))
    coarse = O.tile_features(smooth_surface(250.0), 250.0, rough_baselines_m=(500.0, 1000.0))
    assert fine["slope_med"].shape == coarse["slope_med"].shape == (4, 4)
    inner = (slice(1, 3), slice(1, 3))  # the four tiles whose border cells all have neighbours
    assert np.allclose(fine["slope_med"][inner], coarse["slope_med"][inner], rtol=TOL_RESOLUTION)
    assert np.allclose(fine["relief"][inner], coarse["relief"][inner], rtol=TOL_RESOLUTION)
    assert fine["effective_baseline_m"]["slope_relief_m"] == coarse["effective_baseline_m"]["slope_relief_m"] == 500.0


def test_a_native_cell_coarser_than_the_baseline_reports_the_effective_baseline():
    # Mars MOLA is about 463 m: the 500 m baseline becomes 463 m and the 100 m roughness baseline cannot be met
    assert O.baseline_factor(463.0, 500.0) == 1 and O.effective_baseline_m(463.0, 500.0) == 463.0
    out = O.tile_features(np.random.default_rng(0).normal(size=(40, 40)), 463.0, tile_m=4630.0, stride_m=4630.0)
    assert out["effective_baseline_m"]["rough_a"] == 463.0  # not 100: said openly, not hidden
    assert out["effective_baseline_m"]["slope_relief_m"] == 463.0


# --- NaN rule ----------------------------------------------------------------------------------------------------------------
def test_tile_with_too_many_missing_cells_is_nan_not_partial():
    X, Y = grid(100.0)
    dem = 0.05 * X + 3.0 * np.sin(Y / 400.0)
    tile = (slice(60, 120), slice(60, 120))  # the central tile in 100 m cells
    # 25% of the central tile's cells missing: more than the 20% limit
    dem[60:90, 60:90] = np.nan
    out = O.tile_features(dem, 100.0, max_nan_fraction=0.2)
    for key in ("slope_med", "slope_p90", "relief", "rough_a", "rough_b"):
        assert np.isnan(centre_tile(out[key])), key
    # neighbours are untouched (their own share of missing cells is below the limit or zero)
    assert np.isfinite(out["relief"][0, 0]) and np.isfinite(out["rough_a"][2, 2])
    del tile


def test_tile_below_the_limit_still_gets_a_value_and_a_looser_limit_changes_the_answer():
    X, Y = grid(100.0)
    dem = 0.05 * X + 3.0 * np.sin(Y / 400.0)
    dem[80:86, 80:86] = np.nan  # 36 of 3600 native cells (1%)
    strict = O.tile_features(dem, 100.0, max_nan_fraction=0.0)
    default = O.tile_features(dem, 100.0, max_nan_fraction=0.2)
    assert np.isnan(centre_tile(strict["rough_a"])) and np.isfinite(centre_tile(default["rough_a"]))
    assert np.isfinite(centre_tile(default["relief"])) and np.isfinite(centre_tile(default["slope_med"]))


def test_slope_cells_next_to_a_hole_are_nan():
    dem = np.fromfunction(lambda i, j: 0.1 * j * 500.0, (6, 6))
    dem[3, 3] = np.nan
    slope = O.horn_slope_deg(dem, 500.0)
    assert np.isnan(slope[2:5, 2:5]).all() and np.isfinite(slope[1, 1])


# --- tiling, stride, block mean ----------------------------------------------------------------------------------------------
def test_stride_gives_overlapping_tiles_and_a_nonconforming_stride_is_refused():
    X, Y = grid(100.0, 15000.0)  # a 15 km target window
    dem = 0.03 * X + 0.01 * Y
    out = O.tile_features(dem, 100.0, tile_m=6000.0, stride_m=3000.0)
    assert out["slope_med"].shape == (4, 4)  # (15 - 6) / 3 + 1
    with pytest.raises(ValueError, match="whole number"):
        O.tile_features(dem, 700.0 / 7, tile_m=6000.0, stride_m=3050.0)


def test_block_mean_ignores_nan_and_drops_a_ragged_edge():
    a = np.arange(25, dtype=float).reshape(5, 5)
    b = O.block_mean(a, 2)
    assert b.shape == (2, 2) and b[0, 0] == pytest.approx(np.mean([0, 1, 5, 6]))
    a[0, 0] = np.nan
    assert O.block_mean(a, 2)[0, 0] == pytest.approx(np.mean([1, 5, 6]))
    a[0:2, 0:2] = np.nan
    assert np.isnan(O.block_mean(a, 2)[0, 0])
    with pytest.raises(ValueError):
        O.block_mean(a, 0)


def test_plane_rms_degenerate_cases():
    assert np.isnan(O.plane_rms(np.full((2, 2), np.nan)))
    line = np.full((3, 3), np.nan)
    line[1, :] = [1.0, 2.0, 3.0]  # collinear cells: a plane is not determined
    assert np.isnan(O.plane_rms(line))
