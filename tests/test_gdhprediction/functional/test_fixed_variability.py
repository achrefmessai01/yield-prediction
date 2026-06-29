"""Functional test that the Fixed transform erases temperature variability.

Fixed replaces the outdoor temperature with a single indoor setpoint, so the
per-hour GDH is constant and the cumulative GDH is a straight line: a T2M step,
however large, must produce *no* kink. This is the deliberate opposite of the
Identity/AdaptiveShift tests and pins that the setpoint -- not the weather --
drives the GDH clock.
"""

from typing import Final
from unittest import skip

import numpy as np

from gdhprediction.config import FixedCfg, GdhLocalizedPredictionConfig
from gdhprediction.utils import BrightSkyDataArray
from tests.test_gdhprediction.functional import _support
from tests.test_gdhprediction.functional._support import (
    GDH_BASE,
    IN_FORECAST_DAYS,
    LAT,
    LON,
    WEATHER_HOURS,
    VariabilityTestCase,
)

# Indoor setpoint, comfortably above gdh_base so the constant GDH is positive.
FIXED_TEMP: Final = 18.0


# Localized config using the Fixed transform (indoor == FIXED_TEMP).
FIXED_CONFIG: Final = GdhLocalizedPredictionConfig(
    latitude=LAT,
    longitude=LON,
    gdh_base=GDH_BASE,
    transform=FixedCfg(temperature=FIXED_TEMP),
)


class TestFixedErasesVariability(VariabilityTestCase):
    """A temperature step must leave no trace in the GDH output."""

    T_LOW: Final = 8.0
    T_HIGH: Final = 28.0  # a large step that straddles the setpoint
    STEP_IDX: Final = 100
    DURATION: Final = IN_FORECAST_DAYS

    def _weather_raw(self) -> BrightSkyDataArray:
        return _support.make_weather_raw(
            temperature=_support.step_series(
                self.T_LOW,
                self.T_HIGH,
                at=self.STEP_IDX,
                length=WEATHER_HOURS,
            ),
        )

    def test_temperature_step_produces_no_kink(self) -> None:
        """The cumulative GDH is a straight line; its slope ignores the T2M step."""
        weather_raw = self._weather_raw()
        # Fixed: indoor is the constant setpoint regardless of the outdoor step.
        indoor = np.full(WEATHER_HOURS, FIXED_TEMP, dtype=np.float32)
        _, increments = self.assert_pipeline_matches(
            FIXED_CONFIG,
            weather_raw=weather_raw,
            indoor=indoor,
            duration=self.DURATION,
        )

        fixed_inc = round(FIXED_TEMP - GDH_BASE, 2)  # 18 - 4.5 = 13.5
        # Constant slope everywhere -> no kink despite the T2M step.
        self.assertAllClose(increments, fixed_inc, atol=1.5e-4)
        # Slope is identical on both sides of where the input stepped.
        self.assertAlmostEqual(
            float(increments[self.STEP_IDX - 20]),
            float(increments[self.STEP_IDX + 20]),
            places=6,
        )
        # The slope reflects the setpoint, not either outdoor temperature.
        self.assertNotAlmostEqual(fixed_inc, self.T_LOW - GDH_BASE)
        self.assertNotAlmostEqual(fixed_inc, self.T_HIGH - GDH_BASE)

    @skip("Visual aid: unskip locally to see the GDH stay a straight line.")
    def test_plot_no_kink(self) -> None:
        """Plot the stepped input against the kink-free GDH (no asserts)."""
        import matplotlib.pyplot as plt  # local: keep plotting off the test path

        weather_raw = self._weather_raw()
        prediction = _support.predict(
            FIXED_CONFIG,
            weather_raw=weather_raw,
            climate_raw=_support.make_climate_raw_constant(12.0),
            duration=self.DURATION,
        )
        hours = np.arange(prediction.shape[0])
        n = prediction.shape[0]

        fig, (ax_t, ax_inc, ax_cum) = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        ax_t.step(hours, weather_raw["T2M"][:n], where="post", color="tab:red")
        ax_t.axhline(
            FIXED_TEMP,
            color="tab:blue",
            ls="-.",
            label=f"setpoint {FIXED_TEMP}",
        )
        ax_t.axhline(GDH_BASE, color="grey", ls=":", label=f"gdh_base = {GDH_BASE}")
        ax_t.set_ylabel("T2M (degC)")
        ax_t.legend(loc="upper left")

        ax_inc.step(hours[1:], np.diff(prediction), where="post", color="tab:green")
        ax_inc.set_ylabel("per-hour GDH")

        ax_cum.plot(hours, prediction, color="tab:blue")
        ax_cum.set_ylabel("cumulative GDH")
        ax_cum.set_xlabel("hour from start")

        for ax in (ax_t, ax_inc, ax_cum):
            ax.axvline(self.STEP_IDX, color="tab:orange", ls="--", label="T2M step")
        ax_t.set_title("Fixed transform: T2M step -> no kink (straight GDH)")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
