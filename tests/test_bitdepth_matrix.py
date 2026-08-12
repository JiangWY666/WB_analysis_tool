"""Bit-depth matrix: sensor range, band polarity and saturation must agree.

Most imagers write 10/12/14-bit readings into a 16-bit TIFF. Taking the
container maximum (65535) as the full scale made every such file look dark,
which inverted the band-polarity heuristic and reported 0% saturation for a
fully clipped band.
"""

import numpy as np
import pytest

from wb_analyzer.imageio import blot_from_array, saturation_fraction
from wb_analyzer.quant import compute_lane_profiles, resolve_dark_bands_native
from wb_analyzer.roi import RoiRect

SENSOR_DEPTHS = [
    pytest.param(255.0, np.uint8, 8, id="8bit"),
    pytest.param(1023.0, np.uint16, 16, id="10bit-in-uint16"),
    pytest.param(4095.0, np.uint16, 16, id="12bit-in-uint16"),
    pytest.param(16383.0, np.uint16, 16, id="14bit-in-uint16"),
    pytest.param(65535.0, np.uint16, 16, id="16bit"),
]


def dark_bands_on_film(full_scale: float, dtype: type) -> np.ndarray:
    """Bright film background with a dark band, filling the sensor range."""
    image = np.full((120, 160), full_scale * 0.97)
    image[40:80, 40:120] = full_scale * 0.15
    return np.clip(image, 0, full_scale).astype(dtype)


def light_bands_on_dark(full_scale: float, dtype: type) -> np.ndarray:
    """Dark chemiluminescence background with a bright band."""
    image = np.full((120, 160), full_scale * 0.03)
    image[40:80, 40:120] = full_scale * 0.85
    return np.clip(image, 0, full_scale).astype(dtype)


@pytest.mark.parametrize(("full_scale", "dtype", "storage_depth"), SENSOR_DEPTHS)
def test_white_level_follows_the_sensor_not_the_container(
    full_scale: float, dtype: type, storage_depth: int
) -> None:
    blot = blot_from_array(dark_bands_on_film(full_scale, dtype))
    assert blot.white_level == full_scale
    assert blot.bit_depth == storage_depth


@pytest.mark.parametrize(("full_scale", "dtype", "storage_depth"), SENSOR_DEPTHS)
def test_dark_band_polarity_is_detected_at_every_depth(
    full_scale: float, dtype: type, storage_depth: int
) -> None:
    blot = blot_from_array(dark_bands_on_film(full_scale, dtype))
    assert resolve_dark_bands_native(blot.raw, blot.white_level) is True


@pytest.mark.parametrize(("full_scale", "dtype", "storage_depth"), SENSOR_DEPTHS)
def test_light_band_polarity_is_detected_at_every_depth(
    full_scale: float, dtype: type, storage_depth: int
) -> None:
    blot = blot_from_array(light_bands_on_dark(full_scale, dtype))
    assert resolve_dark_bands_native(blot.raw, blot.white_level) is False


@pytest.mark.parametrize(("full_scale", "dtype", "storage_depth"), SENSOR_DEPTHS)
def test_clipped_band_is_flagged_as_saturated(
    full_scale: float, dtype: type, storage_depth: int
) -> None:
    image = dark_bands_on_film(full_scale, dtype)
    image[50:70, 60:100] = 0  # clipped black
    blot = blot_from_array(image)
    fraction = saturation_fraction(blot.raw, dark_bands=True, white_level=blot.white_level)
    assert fraction == pytest.approx(800 / image.size, abs=1e-6)


@pytest.mark.parametrize(("full_scale", "dtype", "storage_depth"), SENSOR_DEPTHS)
def test_quantification_is_independent_of_the_sensor_range(
    full_scale: float, dtype: type, storage_depth: int
) -> None:
    """Band-to-background contrast is what is measured, so scaling the whole
    image to a different sensor depth must give the same relative answer."""
    blot = blot_from_array(dark_bands_on_film(full_scale, dtype))
    lane = compute_lane_profiles(
        blot.raw,
        RoiRect(10, 10, 140, 100),
        [],
        dark_bands=True,
        white_level=blot.white_level,
    )[0]
    # 40 rows x 80 cols of band at 0.82 of full scale above the film background
    expected = 40 * 80 * full_scale * 0.82
    assert lane.total_density == pytest.approx(expected, rel=0.05)


def test_true_16bit_is_unaffected_by_the_sensor_range_inference() -> None:
    """A genuine 16-bit exposure must still report the full 0..65535 range.

    This is the user-data guard: their chemiluminescence TIFFs sit just below
    65535, and inferring anything narrower would move existing results.
    """
    rng = np.random.default_rng(0)
    image = np.full((200, 200), 65132.0)
    image[80:120, 80:120] -= 20000.0
    image += rng.normal(0.0, 25.0, image.shape)
    blot = blot_from_array(np.clip(image, 0, 65535).astype(np.uint16))
    assert blot.white_level == 65535.0
    assert blot.bit_depth == 16
    assert resolve_dark_bands_native(blot.raw, blot.white_level) is True


def test_non_finite_input_is_rejected_with_a_readable_error() -> None:
    image = np.full((60, 60), 200.0, dtype=np.float32)
    image[30, 30] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        blot_from_array(image)
    image[30, 30] = np.inf
    with pytest.raises(ValueError, match="Inf"):
        blot_from_array(image)
