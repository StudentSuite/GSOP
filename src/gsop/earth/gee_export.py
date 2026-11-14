"""Earth Engine side of the Earth feature build (``gsop s2 earth --source gee``).

STATUS: written against the Earth Engine Python API but **never executed against live Earth Engine**: no
login existed on the build machine when this was written (issue B01). Everything that does not touch
Earth Engine (chunk planning, task ledger, retries, band lists) is unit tested; the Earth Engine
expressions are not. ``gsop check-operators`` (issue B24) and the small-region export (issue B17) are the
first real tests and must pass before any global export.

Every function takes the initialised ``ee`` module as its first argument (``gee_auth.ensure_ee()``), so
this module imports without Earth Engine installed or logged in.

The terrain operator is defined once in ``docs/SPEC.md`` (section Operators) and implemented twice: in
numpy (``operators.py``) and here. Settings come from ``config/features.yaml`` (``operators`` block).
"""

from __future__ import annotations

import math
from typing import Any

from gsop.config import Config

NASADEM_ID = "NASA/NASADEM_HGT/001"
NASADEM_BAND = "elevation"
COPERNICUS_ID = "COPERNICUS/DEM/GLO30_2024_1"
COPERNICUS_BAND = "DEM"

# Horn (1981) weights. dz/dx = sum(KX * z) / (8 * dx); dz/dy = sum(KY * z) / (8 * dy).
HORN_KX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
HORN_KY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]


def lonlat_box_around(lat: float, lon: float, size_km: float) -> list[float]:
    """``[west, south, east, north]`` of a box about ``size_km`` wide and tall centred on a point.

    Plain spherical approximation (111.195 km per degree of latitude); good enough to choose an export
    region. Not used for scoring.
    """
    half_lat = (size_km / 2) / 111.195
    cos_lat = max(math.cos(math.radians(lat)), 0.05)
    half_lon = half_lat / cos_lat
    return [lon - half_lon, max(lat - half_lat, -90.0), lon + half_lon, min(lat + half_lat, 90.0)]


def dem_mosaic(ee: Any, cfg: Config) -> Any:
    """One elevation image: NASADEM inside the configured latitude range, Copernicus GLO-30 outside it.

    Seam rule (documented in SPEC): the switch is a hard cut at the configured latitudes; there is no
    blending. Issue B07 measures whether that creates a step.
    """
    south, north = cfg.features.operators.nasadem_lat_range
    nasa = ee.Image(NASADEM_ID).select(NASADEM_BAND).rename("elevation")
    cop_col = ee.ImageCollection(COPERNICUS_ID).select(COPERNICUS_BAND)
    cop_proj = cop_col.first().select(0).projection()
    cop = cop_col.mosaic().setDefaultProjection(cop_proj).rename("elevation")
    lat = ee.Image.pixelLonLat().select("latitude")
    in_nasa = lat.gte(south).And(lat.lte(north))
    merged = ee.ImageCollection([cop.updateMask(in_nasa.Not()), nasa.updateMask(in_nasa)]).mosaic()
    return merged.setDefaultProjection(nasa.projection()).rename("elevation")


def resample_mean(ee: Any, image: Any, cfg: Config, scale_m: float, max_pixels: int = 4096) -> Any:
    """Block-mean ``image`` to ``scale_m`` metres on the grid CRS (``reduceResolution`` mean)."""
    return (
        image.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=max_pixels)
        .reproject(crs=cfg.features.operators.grid_crs, scale=scale_m)
    )


def horn_slope_deg(ee: Any, dem: Any, cell_m: float) -> Any:
    """Horn slope in degrees on a ``cell_m`` grid. Cells with any missing 3x3 neighbour are masked."""
    kx = ee.Kernel.fixed(3, 3, HORN_KX, 1, 1, False)
    ky = ee.Kernel.fixed(3, 3, HORN_KY, 1, 1, False)
    valid = dem.mask().reduceNeighborhood(reducer=ee.Reducer.min(), kernel=ee.Kernel.square(1, "pixels"))
    filled = dem.unmask(0)
    gx = filled.convolve(kx).divide(8.0 * cell_m)
    gy = filled.convolve(ky).divide(8.0 * cell_m)
    slope = gx.pow(2).add(gy.pow(2)).sqrt().atan().multiply(180.0 / math.pi)
    return slope.updateMask(valid).rename("slope")


def tile_cells(cfg: Config, cell_m: float) -> int:
    ops = cfg.features.operators
    return int(round(ops.tile_m / cell_m)) ** 2


def slope_bands(ee: Any, cfg: Config) -> Any:
    """Bands ``slope_med`` and ``slope_p90`` (degrees) per 6 km tile, from slope at the 500 m baseline.

    DEM -> block mean to 500 m -> Horn slope -> median and 90th percentile over the tile's 12 x 12 cells.
    A tile with more than ``max_nan_fraction`` missing cells is masked (never a partial answer).
    """
    ops = cfg.features.operators
    dem500 = resample_mean(ee, dem_mosaic(ee, cfg), cfg, ops.baseline_m)
    slope = horn_slope_deg(ee, dem500, ops.baseline_m).reproject(crs=ops.grid_crs, scale=ops.baseline_m)
    n = tile_cells(cfg, ops.baseline_m)
    pct = slope.reduceResolution(reducer=ee.Reducer.percentile([50, 90]), maxPixels=n + 16)
    count = slope.reduceResolution(reducer=ee.Reducer.count(), maxPixels=n + 16)
    enough = count.gte(math.ceil((1.0 - ops.max_nan_fraction) * n))
    return pct.rename(["slope_med", "slope_p90"]).updateMask(enough)


def _enough_cells(ee: Any, image: Any, cfg: Config, cell_m: float, max_pixels: int) -> Any:
    """Mask that is 1 where a tile has enough valid cells (more than ``max_nan_fraction`` missing: 0)."""
    n = tile_cells(cfg, cell_m)
    count = image.reduceResolution(reducer=ee.Reducer.count(), maxPixels=max_pixels)
    return count.gte(math.ceil((1.0 - cfg.features.operators.max_nan_fraction) * n))


def relief_band(ee: Any, cfg: Config) -> Any:
    """Band ``relief`` (m): max minus min of the 500 m cells inside the 6 km tile."""
    ops = cfg.features.operators
    dem500 = resample_mean(ee, dem_mosaic(ee, cfg), cfg, ops.baseline_m)
    n = tile_cells(cfg, ops.baseline_m)
    hi = dem500.reduceResolution(reducer=ee.Reducer.max(), maxPixels=n + 16)
    lo = dem500.reduceResolution(reducer=ee.Reducer.min(), maxPixels=n + 16)
    return hi.subtract(lo).rename("relief").updateMask(_enough_cells(ee, dem500, cfg, ops.baseline_m, n + 16))


def roughness_baseline_m(cfg: Config, column: str) -> float:
    """Baseline in metres of ``rough_a`` / ``rough_b``: lives in config/features.yaml, never in code."""
    return float(cfg.features.columns[column].extra["baseline_m"])


def rough_band(ee: Any, cfg: Config, column: str) -> Any:
    """Band ``rough_a`` or ``rough_b`` (m): RMS height after plane removal over the 6 km tile.

    The DEM is block-averaged to the column's baseline B (100 m or 1 km), then inside each tile a plane
    ``z = a + b*x + c*y`` is fitted and removed and the RMS of the residual is taken. On a full square
    grid with tile-centred coordinates ``x`` and ``y`` are orthogonal to each other and to the constant, so
    the least-squares plane has a closed form and no regression is needed::

        mean_sq_resid = E[z^2] - E[z]^2 - E[x z]^2 / E[x^2] - E[y z]^2 / E[y^2],   E[x^2] = (n^2 - 1) / 12

    where ``n`` is the number of cells along a tile side and every ``E`` is a tile mean (computed with
    ``reduceResolution(mean)``). This is exact for a tile with no missing cells; for a tile with some
    missing cells it is an approximation (the numpy operator fits the plane by least squares on the valid
    cells). Tiles with too many missing cells are masked.
    """
    ops = cfg.features.operators
    baseline = roughness_baseline_m(cfg, column)
    n = int(round(ops.tile_m / baseline))
    z = resample_mean(ee, dem_mosaic(ee, cfg), cfg, baseline).select(0).rename("z")
    px = ee.Image.pixelCoordinates(z.projection())
    centre = (n - 1) / 2.0

    def local(band: str) -> Any:  # index of the cell inside its tile, centred; floor-mod so it is never negative
        v = px.select(band)
        return v.subtract(v.divide(n).floor().multiply(n)).subtract(centre)

    stack = ee.Image.cat([z, z.pow(2), local("x").multiply(z), local("y").multiply(z)]).rename(["z", "z2", "xz", "yz"])
    mean = stack.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=n * n + 16)
    ex2 = (n * n - 1) / 12.0
    resid = (
        mean.select("z2").subtract(mean.select("z").pow(2))
        .subtract(mean.select("xz").pow(2).divide(ex2))
        .subtract(mean.select("yz").pow(2).divide(ex2))
    )
    rms = resid.max(0).sqrt().rename(column)
    return rms.updateMask(_enough_cells(ee, z, cfg, baseline, n * n + 16))


MODIS_LST_ID = "MODIS/061/MOD11A2"


def lst_qa_is_good(qc: int) -> bool:
    """MOD11A2 QC rule used for temperature: mandatory-QA bits 0-1 equal 0 (good quality), nothing else.

    (Bits 0-1: 0 good, 1 unreliable or unquantifiable, 2 not produced due to clouds, 3 not produced for other
    reasons; from the catalog page, docs/PROVENANCE.md.) The Earth Engine expression applies the same rule.
    """
    return (int(qc) & 3) == 0


def _lst_composites(ee: Any, cfg: Config) -> Any:
    """8-day composites 2015-2024 with ``temp`` = mean of day and night LST in K, only where BOTH are good."""
    t = cfg.features.temperature
    scale = t.lst_scale_kelvin
    col = (
        ee.ImageCollection(MODIS_LST_ID)
        .filterDate(f"{t.start_year}-01-01", f"{t.end_year + 1}-01-01")
    )

    def to_kelvin(img: Any) -> Any:
        day_ok = img.select("QC_Day").bitwiseAnd(3).eq(0)
        night_ok = img.select("QC_Night").bitwiseAnd(3).eq(0)
        day = img.select("LST_Day_1km").multiply(scale).updateMask(day_ok)
        night = img.select("LST_Night_1km").multiply(scale).updateMask(night_ok)
        mean = day.add(night).divide(2).rename("temp")  # masked where either is masked
        return mean.copyProperties(img, ["system:time_start"])

    return col.map(to_kelvin)


def temperature_bands(ee: Any, cfg: Config) -> Any:
    """Bands ``t_mean``, ``t_coldest`` (K) and ``t_low_obs`` (1 = too few valid observations) at 6 km.

    * Composite temperature = mean of day and night LST of one 8-day MOD11A2 composite, kept only when both
      have QA "good" (day-only values would be biased warm).
    * ``t_mean`` = mean of all valid composites 2015-2024.
    * ``t_coldest`` = the lowest of the 12 calendar-month means (each month averaged over all years), using
      only months with at least ``min_valid_obs_per_month`` valid composites.
    * ``t_low_obs`` = 1 where the number of valid composites is below ``min_valid_obs_total``; the temperature
      bands are masked there (NaN, never a fake number).
    * 1 km values are averaged to the 6 km tile with ``reduceResolution(mean)``.
    """
    t = cfg.features.temperature
    ops = cfg.features.operators
    comps = _lst_composites(ee, cfg)
    proj = comps.first().select(0).projection()
    mean_all = comps.mean().setDefaultProjection(proj)
    n_valid = comps.count().setDefaultProjection(proj)

    monthly = []
    for month in range(1, 13):
        m = comps.filter(ee.Filter.calendarRange(month, month, "month"))
        ok = m.count().gte(t.min_valid_obs_per_month)
        monthly.append(m.mean().updateMask(ok).rename("temp"))
    coldest = ee.ImageCollection.fromImages(monthly).min().setDefaultProjection(proj)

    low_obs = n_valid.lt(t.min_valid_obs_total).rename("t_low_obs")
    stack = ee.Image.cat([mean_all.rename("t_mean"), coldest.rename("t_coldest")])
    cells = int(round(ops.tile_m / 926.0)) ** 2 + 64  # about 6 x 6 MODIS cells, with slack
    t6 = stack.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=cells)
    low6 = low_obs.reduceResolution(reducer=ee.Reducer.max(), maxPixels=cells)
    return t6.updateMask(low6.Not()).addBands(low6.rename("t_low_obs"))


TERRACLIMATE_ID = "IDAHO_EPSCOR/TERRACLIMATE"


def ti_proxy_bands(ee: Any, cfg: Config) -> Any:
    """Bands ``ti_proxy`` (K) and ``ti_proxy_fallback`` (1 = no dry season, median swing used) at 6 km.

    Same definition as ``gsop.ti_proxy`` (numpy reference): swing = day LST minus night LST per 8-day MOD11A2
    composite (both QA good); the dry season is the 3 consecutive cyclic months with the least TerraClimate
    precipitation; ``ti_proxy`` is the mean of those months' mean swings; fallback is the median swing over all
    valid composites. Computed on the MODIS 1 km grid (precipitation is sampled onto it, nearest neighbour),
    then averaged to the 6 km tile (the fallback flag takes the tile maximum).
    """
    t = cfg.features.temperature
    ops = cfg.features.operators
    k = t.dry_months
    years = (f"{t.start_year}-01-01", f"{t.end_year + 1}-01-01")
    col = ee.ImageCollection(MODIS_LST_ID).filterDate(*years)
    proj = col.first().select(0).projection()

    def swing(img: Any) -> Any:
        ok = img.select("QC_Day").bitwiseAnd(3).eq(0).And(img.select("QC_Night").bitwiseAnd(3).eq(0))
        s = img.select("LST_Day_1km").subtract(img.select("LST_Night_1km")).multiply(t.lst_scale_kelvin)
        return s.updateMask(ok).rename("swing").copyProperties(img, ["system:time_start"])

    swings = col.map(swing)
    tc = ee.ImageCollection(TERRACLIMATE_ID).filterDate(*years).select("pr")
    pr_m = [tc.filter(ee.Filter.calendarRange(m, m, "month")).mean().multiply(t.precip_scale).rename("pr") for m in range(1, 13)]
    sw_m = [swings.filter(ee.Filter.calendarRange(m, m, "month")).mean().rename("swing") for m in range(1, 13)]

    sums = [sum_images(ee, [pr_m[(s + j) % 12] for j in range(k)]).rename("sum") for s in range(12)]
    totals = ee.Image.cat(sums)  # 12 bands: window totals for start month 0..11
    start = totals.toArray().arrayArgmin().arrayGet([0]).rename("start")
    driest = ee.ImageCollection.fromImages(sums).min().rename("driest")
    annual = sum_images(ee, pr_m)
    defined = driest.lt(annual.divide(12.0).multiply(k * t.dry_season_ratio_max)).Or(annual.lte(1e-6))

    in_window = []
    for m in range(12):
        # month m (0-based) is in the dry window iff (m - start) mod 12 < k  (+12 keeps the value positive)
        member = start.multiply(-1).add(m + 12).mod(12).lt(k)
        in_window.append(sw_m[m].updateMask(member))
    dry_swing = ee.ImageCollection.fromImages(in_window).mean()
    median_all = swings.median()
    use_dry = defined.And(dry_swing.mask())
    ti = median_all.where(use_dry, dry_swing).rename("ti_proxy")
    fallback = use_dry.Not().rename("ti_proxy_fallback")
    cells = int(round(ops.tile_m / 926.0)) ** 2 + 64
    ti6 = ti.setDefaultProjection(proj).reduceResolution(reducer=ee.Reducer.mean(), maxPixels=cells)
    fb6 = fallback.setDefaultProjection(proj).reduceResolution(reducer=ee.Reducer.max(), maxPixels=cells)
    return ti6.addBands(fb6)


def sum_images(ee: Any, images: list) -> Any:
    total = images[0]
    for img in images[1:]:
        total = total.add(img)
    return total


def water_index_status(cfg: Config) -> tuple[str, str]:
    """('available' | 'unavailable', reason). ``unavailable`` while no permafrost asset is configured (B04)."""
    w = cfg.features.water_index
    if not w.permafrost_asset:
        return (
            "unavailable",
            "no permafrost-probability layer is configured (features.yaml water_index.permafrost_asset is null): "
            "none exists in the Earth Engine catalog; decision needed from Anay",
        )
    return "available", ""


def aridity_dryness_image(ee: Any, cfg: Config) -> Any:
    """Dryness 0..1 (1 = hyperarid) from TerraClimate: ``1 - clip(sum(pr) / sum(pet) / ai_cap, 0, 1)``.

    Annual sums over the climatology years; ``pet`` is scaled by 0.1 (catalog page: mm, scale 0.1) and ``pr`` by
    ``temperature.precip_scale`` (the page lists no unit or scale for ``pr``: UNVERIFIED).
    """
    t = cfg.features.temperature
    cap = cfg.features.water_index.aridity_ai_cap
    years = (f"{t.start_year}-01-01", f"{t.end_year + 1}-01-01")
    tc = ee.ImageCollection(TERRACLIMATE_ID).filterDate(*years)
    n_years = t.end_year - t.start_year + 1
    pr = tc.select("pr").sum().multiply(t.precip_scale).divide(n_years)
    pet = tc.select("pet").sum().multiply(0.1).divide(n_years)
    ai = pr.divide(pet.updateMask(pet.gt(0)))
    return ai.divide(cap).clamp(0, 1).multiply(-1).add(1).rename("dryness")


def water_index_bands(ee: Any, cfg: Config) -> Any:
    """Band ``water_index`` (0-1) at 6 km, plus the diagnostic band ``dryness``.

    Raises ``RuntimeError`` while the permafrost layer is unavailable: the orchestration then logs the reason and
    leaves ``water_index`` out, so the column is NaN and the missing-data gate acts. Nothing is substituted.
    """
    status, reason = water_index_status(cfg)
    if status != "available":
        raise RuntimeError(f"water_index unavailable: {reason}")
    w = cfg.features.water_index
    ops = cfg.features.operators
    perm = ee.Image(w.permafrost_asset).select(0).rename("permafrost")
    if w.permafrost_outside_domain == "zero":
        perm = perm.unmask(0)
    dry = aridity_dryness_image(ee, cfg)
    wi = perm.add(dry).divide(2).rename("water_index")
    cells = int(round(ops.tile_m / 1000.0)) ** 2 + 64
    return wi.addBands(dry).reduceResolution(reducer=ee.Reducer.mean(), maxPixels=cells)


MERRA2_AER_ID = "NASA/GSFC/MERRA/aer/2"


def dust_bands(ee: Any, cfg: Config) -> Any:
    """Band ``dust_index_raw``: annual-mean MERRA-2 dust extinction AOT (550 nm, unitless) at 6 km.

    Only the 12:00 UTC hourly value of each day is used (one image per day instead of 24) to keep the reduction
    small; dust optical thickness changes slowly within a day. MERRA-2 cells are 69375 x 55000 m, so every 6 km
    tile inside a cell gets the cell's value (nearest neighbour): coarse by design, see ``gsop.dust``. Scaling
    to 0..1 happens at build time (B18) with data percentiles.
    """
    t = cfg.features.temperature
    col = (
        ee.ImageCollection(MERRA2_AER_ID)
        .filterDate(f"{t.start_year}-01-01", f"{t.end_year + 1}-01-01")
        .filter(ee.Filter.calendarRange(12, 12, "hour"))
        .select("DUEXTTAU")
    )
    return col.mean().rename("dust_index_raw")


ASTER_GED_ID = "NASA/ASTER_GED/AG100_003"


def silica_bands(ee: Any, cfg: Config) -> Any:
    """Band ``silica_index``: ASTER Quartz Index ``(e11 * e11) / (e10 * e12)``, averaged to the 6 km tile.

    ASTER GED v003 (2000-2008, 111.32 m). Water pixels (``land_water_map`` = 2) and non-positive emissivities are
    masked. Higher = more silica-rich. Gaps in ASTER GED become masked pixels, hence NaN tiles; see SPEC.
    """
    scale = cfg.features.silica_index.aster_emissivity_scale
    ops = cfg.features.operators
    img = ee.Image(ASTER_GED_ID)
    e10, e11, e12 = (img.select(f"emissivity_band{n}").multiply(scale) for n in (10, 11, 12))
    ok = e10.gt(0).And(e11.gt(0)).And(e12.gt(0)).And(img.select("land_water_map").eq(1))
    qi = e11.multiply(e11).divide(e10.multiply(e12)).updateMask(ok).rename("silica_index")
    cells = int(round(ops.tile_m / 111.32)) ** 2 + 128
    return qi.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=cells)
