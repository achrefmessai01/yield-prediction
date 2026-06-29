"""Functional tests that temperature variability propagates into the GDH output.

Unlike ``tests.test_gdhprediction.test_apis`` (which mocks the whole GdhData
objects and injects pre-computed cumulative GDH to exercise the stitch), these
tests run the real ``transform -> max(0, T - base) -> cumsum -> crop -> stitch``
pipeline end to end via the helpers in ``_support`` (which patch only the raw
fetch and force a cache miss).

With the Identity transform the indoor temperature equals the forecast T2M, so
the per-hour GDH equals ``max(0, T2M - gdh_base)`` and ``np.diff`` of the output
recovers exactly the injected increment series. A temperature step must then
appear as a slope change at the right hour -- the property under test.
"""

from typing import Final
from unittest import skip

import numpy as np

from gdhprediction.config import GdhLocalizedPredictionConfig, IdentityCfg
from gdhprediction.utils import BrightSkyDataArray
from tests.test_gdhprediction.functional import _support
from tests.test_gdhprediction.functional._support import (
    GDH_BASE,
    IN_FORECAST_DAYS,
    LAT,
    LON,
    MONOTONIC_TOL,
    WEATHER_HOURS,
    VariabilityTestCase,
)

# Localized config using the Identity transform (indoor T2M == forecast T2M).
IDENTITY_CONFIG: Final = GdhLocalizedPredictionConfig(
    latitude=LAT,
    longitude=LON,
    gdh_base=GDH_BASE,
    transform=IdentityCfg(),
)


class TestIdentityVariabilityPropagation(VariabilityTestCase):
    """Inject a temperature step and assert it reaches the cumulative GDH output."""

    T_LOW: Final = 10.0
    T_HIGH: Final = 20.0
    STEP_IDX: Final = 100
    DURATION: Final = 28  # default horizon in days

    def _weather_raw(self) -> BrightSkyDataArray:
        return _support.make_weather_raw(
            temperature=_support.step_series(
                self.T_LOW,
                self.T_HIGH,
                at=self.STEP_IDX,
                length=WEATHER_HOURS,
            ),
        )

    def test_weather_only_step_recovers_increment_series(self) -> None:
        """Recover the injected step from a horizon fully inside the forecast.

        A 9-day horizon fits within the 240 h forecast, so the output is purely
        transform-derived: the cumulative GDH re-anchors to zero and its hourly
        differences reproduce ``max(0, T2M - base)`` exactly.
        """
        duration = IN_FORECAST_DAYS
        weather_raw = self._weather_raw()
        # Identity: indoor temperature is the forecast T2M unchanged.
        _, increments = self.assert_pipeline_matches(
            IDENTITY_CONFIG,
            weather_raw=weather_raw,
            indoor=weather_raw["T2M"],
            duration=duration,
        )

        # The step survives as a slope change between the two plateaus.
        low_inc = round(self.T_LOW - GDH_BASE, 2)
        high_inc = round(self.T_HIGH - GDH_BASE, 2)
        self.assertAllClose(increments[: self.STEP_IDX - 10], low_inc, atol=1.5e-4)
        self.assertAllClose(increments[self.STEP_IDX + 10 :], high_inc, atol=1.5e-4)
        # Variability is genuinely visible, not a flat line.
        self.assertGreater(high_inc, low_inc)

    def test_full_horizon_step_survives_stitch_and_climate_level_propagates(
        self,
    ) -> None:
        """Carry the step through the weather/climate seam over the full horizon.

        Over 28 days the weather step must remain visible in the forecast
        portion, the seam must stay continuous, and the climate leg's injected
        temperature must drive a constant slope past the seam -- so the stitched
        output carries three distinct, input-driven slopes.
        """
        duration = self.DURATION  # weather (240 h) < 672 h, so climate extends
        duration_hours = duration * 24
        t_climate = 12.0
        weather_raw = self._weather_raw()
        prediction = _support.predict(
            IDENTITY_CONFIG,
            weather_raw=weather_raw,
            climate_raw=_support.make_climate_raw_constant(t_climate),
            duration=duration,
        )
        self.assertEqual(prediction.shape, (duration_hours,))
        increments = np.diff(prediction)

        # Weather portion (first 240 h): identical to the weather-only case.
        expected_weather = _support.expected_prediction(
            weather_raw["T2M"],
            duration_hours=WEATHER_HOURS,
        )
        self.assertAllClose(
            prediction[:WEATHER_HOURS],
            expected_weather,
            atol=1.5e-4,
        )

        # Seam continuity: the climate leg re-anchors onto the weather tail, so
        # the first climate sample duplicates the last weather value.
        seam = WEATHER_HOURS
        self.assertAlmostEqual(
            float(prediction[seam]),
            float(expected_weather[-1]),
            places=4,
        )

        # Climate level propagates as a constant slope past the seam.
        climate_inc = round(t_climate - GDH_BASE, 2)
        post_seam = increments[seam:]
        self.assertAllClose(post_seam, climate_inc, atol=1.5e-4)

        # The three slopes are genuinely distinct, so every kink is visible. The
        # weather plateaus and the climate slope are each pinned exactly above; this
        # just records that the scenario constants give three different slopes.
        low_inc = round(self.T_LOW - GDH_BASE, 2)
        high_inc = round(self.T_HIGH - GDH_BASE, 2)
        self.assertNotAlmostEqual(climate_inc, low_inc)
        self.assertNotAlmostEqual(climate_inc, high_inc)

        # Cumulative GDH is monotonic non-decreasing throughout.
        self.assertTrue(np.all(increments >= MONOTONIC_TOL))

    @skip("Visual aid: unskip locally to eyeball the temperature->GDH kink.")
    def test_plot_step_to_gdh_kink(self) -> None:
        """Plot the injected temperature against the resulting GDH (no asserts).

        Unskip and run this single test locally to see the step propagate: the
        input temperature jumps, the per-hour GDH jumps with it, and the
        cumulative GDH bends at the weather step and again at the climate seam.
        Saves a PNG to the scratch dir and opens an interactive window.
        """
        import matplotlib.pyplot as plt  # local: keep plotting off the test path

        weather_raw = self._weather_raw()
        t_climate = 12.0
        prediction = _support.predict(
            IDENTITY_CONFIG,
            weather_raw=weather_raw,
            climate_raw=_support.make_climate_raw_constant(t_climate),
            duration=self.DURATION,
        )
        hours = np.arange(prediction.shape[0])
        seam = WEATHER_HOURS

        # Reconstruct the temperature the prediction "saw": the weather step for
        # the forecast window, then the constant climate level past the seam.
        temperature = np.empty_like(prediction)
        temperature[:seam] = weather_raw["T2M"][:seam]
        temperature[seam:] = t_climate

        fig, (ax_t, ax_inc, ax_cum) = plt.subplots(
            3,
            1,
            sharex=True,
            figsize=(10, 8),
        )
        ax_t.step(hours, temperature, where="post", color="tab:red")
        ax_t.axhline(GDH_BASE, color="grey", ls=":", label=f"gdh_base = {GDH_BASE}")
        ax_t.set_ylabel("T2M (degC)")
        ax_t.legend(loc="upper left")

        ax_inc.step(hours[1:], np.diff(prediction), where="post", color="tab:green")
        ax_inc.set_ylabel("per-hour GDH")

        ax_cum.plot(hours, prediction, color="tab:blue")
        ax_cum.set_ylabel("cumulative GDH")
        ax_cum.set_xlabel("hour from start")

        for ax in (ax_t, ax_inc, ax_cum):
            ax.axvline(self.STEP_IDX, color="tab:orange", ls="--", label="weather step")
            ax.axvline(seam, color="tab:purple", ls="--", label="climate seam")
        ax_t.set_title("Identity transform: temperature step -> GDH kink")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
