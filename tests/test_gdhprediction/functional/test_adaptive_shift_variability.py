"""Functional tests that AdaptiveShift turns temperature *and* sunshine into kinks.

Companion to ``test_identity_variability``: same end-to-end approach via the
``_support`` helpers (patch the raw fetch, force a cache miss, run the real
``transform -> max(0, T - base) -> cumsum -> crop`` pipeline) but with the
AdaptiveShift transform.

The injected weather holds two independent steps at different hours: a step in
T2M and a step in shortwave irradiance. Kept inside the comfort band
(``min <= passive <= max``) AdaptiveShift reduces to
``indoor = T2M + (solar_factor / base_heat_loss) * solar``, so each step is a
distinct, exactly-predictable slope change in the cumulative GDH: the first kink
is driven by temperature, the second purely by sunshine.
"""

from typing import Final
from unittest import skip

import numpy as np

from gdhprediction.config import (
    AdaptiveCoefficients,
    AdaptiveShiftCfg,
    GdhLocalizedPredictionConfig,
)
from gdhprediction.utils import BrightSkyDataArray, Float32
from tests.test_gdhprediction.functional import _support
from tests.test_gdhprediction.functional._support import (
    GDH_BASE,
    IN_FORECAST_DAYS,
    LAT,
    LON,
    WEATHER_HOURS,
    VariabilityTestCase,
)

# AdaptiveShift coefficients. base_heat_loss/solar_factor give a solar coupling
# of k = 10 / 500 = 0.02 degC per W/m^2. The comfort band is wide enough that
# every passive temperature below stays inside it, so neither the heating nor
# the venting branch ever fires -- isolating the linear temperature/solar kinks.
BASE_HEAT_LOSS: Final = 500.0
SOLAR_FACTOR: Final = 10.0
SOLAR_COUPLING: Final = SOLAR_FACTOR / BASE_HEAT_LOSS  # 0.02 degC per W/m^2
TEMP_RANGE: Final = (5.0, 30.0)
VENTILATION: Final = 5.0
HEATING: Final = 50_000.0


# Localized config using AdaptiveShift via its coefficients route.
ADAPTIVE_CONFIG: Final = GdhLocalizedPredictionConfig(
    latitude=LAT,
    longitude=LON,
    gdh_base=GDH_BASE,
    transform=AdaptiveShiftCfg(
        params=AdaptiveCoefficients(
            base_heat_loss=BASE_HEAT_LOSS,
            solar_factor=SOLAR_FACTOR,
            ventilation=VENTILATION,
            heating=HEATING,
            temperature_range=TEMP_RANGE,
        ),
    ),
)


class TestAdaptiveShiftVariabilityPropagation(VariabilityTestCase):
    """A temperature step and a sunshine step must each bend the GDH curve."""

    DURATION: Final = IN_FORECAST_DAYS

    # Two well-separated steps, both inside the 216 h returned window.
    TEMP_STEP_IDX: Final = 80
    SOLAR_STEP_IDX: Final = 160
    T_LOW: Final = 10.0
    T_HIGH: Final = 18.0
    S_LOW: Final = 0.0
    S_HIGH: Final = 300.0

    def _weather_raw(self) -> BrightSkyDataArray:
        return _support.make_weather_raw(
            temperature=_support.step_series(
                self.T_LOW,
                self.T_HIGH,
                at=self.TEMP_STEP_IDX,
                length=WEATHER_HOURS,
            ),
            irradiance=_support.step_series(
                self.S_LOW,
                self.S_HIGH,
                at=self.SOLAR_STEP_IDX,
                length=WEATHER_HOURS,
            ),
        )

    @staticmethod
    def _indoor(weather_raw: BrightSkyDataArray) -> Float32:
        """Comfort-band inner temperature: T2M + k*solar."""
        t2m = weather_raw["T2M"].astype(np.float32)
        solar = weather_raw["ALLSKY_SFC_SW_DWN"].astype(np.float32)
        return t2m + np.float32(SOLAR_COUPLING) * solar

    def test_temperature_and_sunshine_steps_make_two_distinct_kinks(self) -> None:
        """Recover three slopes: a baseline, a temperature kink, a sunshine kink.

        In the comfort band indoor == T2M + k*solar, so per-hour GDH is exactly
        ``max(0, T2M + k*solar - base)``. The two input steps land on separate
        hours, so the cumulative GDH bends twice -- once from temperature, once
        from irradiance -- and ``np.diff`` recovers all three plateaus.
        """
        weather_raw = self._weather_raw()
        # Comfort-band AdaptiveShift: indoor == T2M + k*solar.
        indoor = self._indoor(weather_raw)
        _, increments = self.assert_pipeline_matches(
            ADAPTIVE_CONFIG,
            weather_raw=weather_raw,
            indoor=indoor,
            duration=self.DURATION,
            atol=1.5e-3,
        )

        # The three plateaus, derived straight from the inputs.
        inc_base = round(self.T_LOW - GDH_BASE, 2)  # 10 - 4.5 = 5.5
        inc_warm = round(self.T_HIGH - GDH_BASE, 2)  # 18 - 4.5 = 13.5
        inc_sunny = round(  # 18 + 0.02*300 - 4.5 = 19.5
            self.T_HIGH + SOLAR_COUPLING * self.S_HIGH - GDH_BASE,
            2,
        )
        # Each plateau holds well clear of the two transitions.
        self.assertAllClose(
            increments[: self.TEMP_STEP_IDX - 10],
            inc_base,
            atol=1.5e-3,
        )
        mid = increments[self.TEMP_STEP_IDX + 10 : self.SOLAR_STEP_IDX - 10]
        self.assertAllClose(mid, inc_warm, atol=1.5e-3)
        self.assertAllClose(
            increments[self.SOLAR_STEP_IDX + 10 :],
            inc_sunny,
            atol=1.5e-3,
        )

        # Kink 1 is temperature-only: the slope jump equals the temperature step.
        self.assertAlmostEqual(inc_warm - inc_base, self.T_HIGH - self.T_LOW, places=4)
        # Kink 2 is sunshine-only (T2M unchanged across it): the slope jump equals
        # the solar coupling times the irradiance step.
        self.assertAlmostEqual(
            inc_sunny - inc_warm,
            SOLAR_COUPLING * (self.S_HIGH - self.S_LOW),
            places=4,
        )
        # The two kinks are genuinely distinct and both present.
        self.assertGreater(inc_warm, inc_base)
        self.assertGreater(inc_sunny, inc_warm)

    def test_stays_in_comfort_band_no_heating_or_venting(self) -> None:
        """Guard the premise: every passive temperature lands inside the band.

        If a chosen input pushed ``passive`` below ``min`` or above ``max`` the
        heating/venting branches would fire and the clean two-kink picture would
        break, so pin that the test inputs keep AdaptiveShift in its linear arm.
        """
        passive = self._indoor(self._weather_raw())
        min_temp, max_temp = TEMP_RANGE
        self.assertGreaterEqual(float(passive.min()), min_temp)
        self.assertLessEqual(float(passive.max()), max_temp)

    @skip("Visual aid: unskip locally to eyeball the temperature and sunshine kinks.")
    def test_plot_temperature_and_sunshine_kinks(self) -> None:
        """Plot the two inputs against the resulting GDH (no asserts).

        Unskip and run this single test locally to watch the cumulative GDH bend
        once where T2M steps and again where irradiance steps. Saves a PNG to the
        temp dir and opens an interactive window.
        """
        import matplotlib.pyplot as plt  # local: keep plotting off the test path

        weather_raw = self._weather_raw()
        prediction = _support.predict(
            ADAPTIVE_CONFIG,
            weather_raw=weather_raw,
            climate_raw=_support.make_climate_raw_constant(12.0),
            duration=self.DURATION,
        )
        hours = np.arange(prediction.shape[0])
        n = prediction.shape[0]

        fig, (ax_t, ax_s, ax_inc, ax_cum) = plt.subplots(
            4,
            1,
            sharex=True,
            figsize=(10, 9),
        )
        ax_t.step(hours, weather_raw["T2M"][:n], where="post", color="tab:red")
        ax_t.axhline(GDH_BASE, color="grey", ls=":", label=f"gdh_base = {GDH_BASE}")
        ax_t.set_ylabel("T2M (degC)")
        ax_t.legend(loc="upper left")

        ax_s.step(
            hours,
            weather_raw["ALLSKY_SFC_SW_DWN"][:n],
            where="post",
            color="tab:orange",
        )
        ax_s.set_ylabel("irradiance (W/m^2)")

        ax_inc.step(hours[1:], np.diff(prediction), where="post", color="tab:green")
        ax_inc.set_ylabel("per-hour GDH")

        ax_cum.plot(hours, prediction, color="tab:blue")
        ax_cum.set_ylabel("cumulative GDH")
        ax_cum.set_xlabel("hour from start")

        for ax in (ax_t, ax_s, ax_inc, ax_cum):
            ax.axvline(self.TEMP_STEP_IDX, color="tab:red", ls="--", label="temp step")
            ax.axvline(
                self.SOLAR_STEP_IDX,
                color="tab:orange",
                ls="--",
                label="sun step",
            )
        ax_t.set_title("AdaptiveShift: temperature kink + sunshine kink")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
