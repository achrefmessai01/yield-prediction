"""Functional tests that Constrained clips temperature into its range.

Constrained sets ``indoor = clip(T2M, min, max)``. So temperature variability
propagates one-to-one only *inside* ``[min, max]``: below ``min`` and above
``max`` the indoor temperature saturates at the boundary, the GDH slope goes
flat, and further outdoor changes leave no kink. This exercises all three
regimes -- inside, on the boundary, and outside -- in one staircase.
"""

from typing import Final
from unittest import skip

import numpy as np

from gdhprediction.config import ConstrainedCfg, GdhLocalizedPredictionConfig
from gdhprediction.utils import BrightSkyDataArray, Float64
from tests.test_gdhprediction.functional import _support
from tests.test_gdhprediction.functional._support import (
    GDH_BASE,
    IN_FORECAST_DAYS,
    LAT,
    LON,
    WEATHER_HOURS,
    VariabilityTestCase,
)

# Clip range, both ends above gdh_base so every clamped GDH stays positive.
C_MIN: Final = 8.0
C_MAX: Final = 20.0


# Localized config using the Constrained transform (clip into [C_MIN, C_MAX]).
CONSTRAINED_CONFIG: Final = GdhLocalizedPredictionConfig(
    latitude=LAT,
    longitude=LON,
    gdh_base=GDH_BASE,
    transform=ConstrainedCfg(temperature_range=(C_MIN, C_MAX)),
)


def _piecewise(edges: tuple[int, ...], values: tuple[float, ...]) -> Float64:
    """Build a WEATHER_HOURS staircase: ``values[i]`` on ``[edges[i], edges[i+1])``."""
    bounds = (0, *edges, WEATHER_HOURS)
    series = np.empty(WEATHER_HOURS, dtype=np.float64)
    for value, lo, hi in zip(values, bounds[:-1], bounds[1:], strict=True):
        series[lo:hi] = value
    return series


class TestConstrainedClamping(VariabilityTestCase):
    """Kinks appear only inside the range; outside the slope is clamped flat."""

    DURATION: Final = IN_FORECAST_DAYS

    # Staircase visiting every regime, each segment held long enough to read a
    # clean slope. Outdoor temps: below min, inside-low, inside-high, above max,
    # and higher-still above max (the last step must NOT add a kink).
    EDGES: Final = (50, 100, 150, 200)
    BELOW: Final = 4.0  # < C_MIN  -> clamps up to C_MIN
    INSIDE_LO: Final = 12.0  # in range
    INSIDE_HI: Final = 16.0  # in range
    ABOVE: Final = 25.0  # > C_MAX -> clamps down to C_MAX
    ABOVE_MORE: Final = 35.0  # >> C_MAX -> still clamps to C_MAX (no new kink)

    def _staircase_raw(self) -> BrightSkyDataArray:
        temperature = _piecewise(
            self.EDGES,
            (self.BELOW, self.INSIDE_LO, self.INSIDE_HI, self.ABOVE, self.ABOVE_MORE),
        )
        return _support.make_weather_raw(temperature=temperature)

    def test_kinks_inside_range_flat_when_clamped(self) -> None:
        """Walk the staircase: clamp, kink in, kink inside, clamp, no kink.

        The recovered per-hour GDH must match ``max(0, clip(T2M) - base)`` and
        show: a clamped-low plateau, a 1:1 kink entering the range, a 1:1 kink
        inside it, a boundary kink on saturation, and *no* kink for a further
        rise once already above the maximum.
        """
        weather_raw = self._staircase_raw()
        duration_hours = self.DURATION * 24
        # Constrained: indoor is the outdoor temperature clipped into the range.
        indoor = np.clip(weather_raw["T2M"].astype(np.float32), C_MIN, C_MAX)
        _, increments = self.assert_pipeline_matches(
            CONSTRAINED_CONFIG,
            weather_raw=weather_raw,
            indoor=indoor,
            duration=self.DURATION,
        )

        # Per-phase slopes, read away from the transitions.
        e0, e1, e2, e3 = self.EDGES
        slope_below = float(increments[e0 // 2])
        slope_inside_lo = float(increments[(e0 + e1) // 2])
        slope_inside_hi = float(increments[(e1 + e2) // 2])
        slope_above = float(increments[(e2 + e3) // 2])
        slope_above_more = float(increments[(e3 + duration_hours) // 2])

        base = GDH_BASE
        # Below the range: clamped up to C_MIN, NOT the (sub-base) outdoor temp.
        self.assertAlmostEqual(slope_below, max(0.0, C_MIN - base), places=4)  # 3.5
        # would be 0 if the sub-base outdoor temp were used instead of the clamp
        self.assertNotAlmostEqual(slope_below, max(0.0, self.BELOW - base))
        # Inside the range: the slope equals the actual temperature minus base.
        self.assertAlmostEqual(slope_inside_lo, self.INSIDE_LO - base, places=4)  # 7.5
        self.assertAlmostEqual(slope_inside_hi, self.INSIDE_HI - base, places=4)  # 11.5
        # Above the range: clamped down to C_MAX, NOT the outdoor temp.
        self.assertAlmostEqual(slope_above, max(0.0, C_MAX - base), places=4)  # 15.5
        self.assertNotAlmostEqual(slope_above, self.ABOVE - base)

        # Kink entering the range is real (clamp -> inside).
        self.assertGreater(slope_inside_lo, slope_below)
        # Kink *inside* the range propagates the outdoor step one-to-one.
        self.assertAlmostEqual(
            slope_inside_hi - slope_inside_lo,
            self.INSIDE_HI - self.INSIDE_LO,
            places=4,
        )
        # Boundary kink on saturation (inside -> above).
        self.assertGreater(slope_above, slope_inside_hi)
        # Outside the range: a further rise (ABOVE -> ABOVE_MORE) adds NO kink.
        self.assertAlmostEqual(slope_above_more, slope_above, places=6)

    def test_boundary_values_pass_through_unclamped(self) -> None:
        """T2M stepping between exactly C_MIN and exactly C_MAX clips to itself.

        ``np.clip`` is inclusive, so boundary values are returned unchanged and
        the step propagates one-to-one -- the edges belong to the active range.
        """
        temperature = _support.step_series(C_MIN, C_MAX, at=100, length=WEATHER_HOURS)
        weather_raw = _support.make_weather_raw(temperature=temperature)
        # Constrained: indoor is the outdoor temperature clipped into the range.
        indoor = np.clip(weather_raw["T2M"].astype(np.float32), C_MIN, C_MAX)
        _, increments = self.assert_pipeline_matches(
            CONSTRAINED_CONFIG,
            weather_raw=weather_raw,
            indoor=indoor,
            duration=self.DURATION,
        )

        self.assertAllClose(increments[:80], C_MIN - GDH_BASE, atol=1.5e-4)  # 3.5
        self.assertAllClose(increments[120:], C_MAX - GDH_BASE, atol=1.5e-4)  # 15.5
        # The full range width propagates as the kink: no clamping at the edges.
        self.assertAlmostEqual(
            float(increments[120]) - float(increments[80]),
            C_MAX - C_MIN,
            places=4,
        )

    @skip("Visual aid: unskip locally to see kinks inside the range, flats outside.")
    def test_plot_clamping(self) -> None:
        """Plot the staircase against the clamped GDH (no asserts)."""
        import matplotlib.pyplot as plt  # local: keep plotting off the test path

        weather_raw = self._staircase_raw()
        prediction = _support.predict(
            CONSTRAINED_CONFIG,
            weather_raw=weather_raw,
            climate_raw=_support.make_climate_raw_constant(12.0),
            duration=self.DURATION,
        )
        hours = np.arange(prediction.shape[0])
        n = prediction.shape[0]

        fig, (ax_t, ax_inc, ax_cum) = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        ax_t.step(hours, weather_raw["T2M"][:n], where="post", color="tab:red")
        ax_t.axhspan(C_MIN, C_MAX, color="tab:blue", alpha=0.1, label="clip range")
        ax_t.axhline(GDH_BASE, color="grey", ls=":", label=f"gdh_base = {GDH_BASE}")
        ax_t.set_ylabel("T2M (degC)")
        ax_t.legend(loc="upper left")

        ax_inc.step(hours[1:], np.diff(prediction), where="post", color="tab:green")
        ax_inc.set_ylabel("per-hour GDH")

        ax_cum.plot(hours, prediction, color="tab:blue")
        ax_cum.set_ylabel("cumulative GDH")
        ax_cum.set_xlabel("hour from start")

        for ax in (ax_t, ax_inc, ax_cum):
            for edge in self.EDGES:
                ax.axvline(edge, color="tab:orange", ls="--")
        ax_t.set_title("Constrained: kinks inside the range, flat when clamped")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
