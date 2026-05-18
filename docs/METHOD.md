# Methodology

## Thermal time model

Strawberry maturation rate is modelled by hourly Growth Degree Hours (GDH):

```
GDH(t) = max(T(t) − T_base, 0)
CUM_GDH(t) = Σ_{τ ≤ t} GDH(τ)
```

with `T_base ≈ 4.5 °C` (configurable). Each maturation stage maps to a CUM_GDH range; the predicted harvest schedule is read off the CUM_GDH timeline at the configured stage thresholds.

## Hourly temperature forecast

The 28–30 day temperature horizon is constructed by splicing two sources:

1. **Short range — numerical weather forecast.** Hourly forecast for ~10 days from a DWD-derived service. Treated as deterministic.
2. **Long range — historical hourly climatology.** ~20 years of historical hourly observations at the grid point closest to the site. Per hour-of-year, the empirical 5 / 25 / 50 / 75 / 95 percentiles are computed across years. The median is the central estimate; the percentile spread is the non-parametric uncertainty band.

The calendar is normalised to a 365-day year (Feb 29 is dropped); leap-year requests are shifted by one day after February to keep the per-hour-of-year alignment.

## Greenhouse heating model

Outdoor temperature is converted to inside-canopy temperature by a simple offset:

- **Identity**: `T_in = T_out` (open field).
- **Shift**: `T_in = T_out + Δ` (passive greenhouse, constant offset).
- **Radiation shift**: `T_in = T_out + Δ + k · R_↓` (offset scaled by incident shortwave radiation; first-order proxy for solar gain).

Selecting and parameterising this transform is how the model adapts to greenhouse type.

## Auxiliary derivations

- **Shortwave radiation from sunshine duration.** Where direct irradiance is unavailable, hourly surface shortwave is estimated from sunshine-minutes via the Ångström–Prescott parameterisation (FAO-56):
  `R_↓ = R_a · (a_s + b_s · n/N)`, with `R_a` the extraterrestrial flux computed from latitude, day-of-year, and solar hour-angle, and `n/N` the fractional sunshine in the hour.
- **Relative humidity from dew point.** When RH is not directly reported, it is reconstructed from temperature and dew point via the Magnus equation.

## Splicing weather and climatology

The two legs are concatenated into a single hourly CUM_GDH timeline:

1. Compute hourly GDH on each leg independently using the same transform and `T_base`.
2. Take the cumulative sum within each leg.
3. Anchor the climatology median to the last value of the weather cumulative sum (subtract the climatology value at the seam, add the weather value). The percentile bands are translated by the same offset.

The seam is placed at the hour just after the last weather record, in climatology's hour-of-year index.

## Aggregation

The native output is hourly CUM_GDH (median + percentile envelope). Daily output is obtained by sampling the last hour of each local day.

## From CUM_GDH to harvest (planned)

Cameras classify a sample of fruits into seven maturation stages. Each stage corresponds to a distribution over remaining CUM_GDH-to-harvest (default: normal, configurable). Sample counts are extrapolated to whole-greenhouse counts by the camera sampling weights. Convolving each stage's residual-GDH distribution with the forecast CUM_GDH(t) yields an expected harvest-rate curve over the horizon, with uncertainty propagated from the climatology percentile band.

## Deferred ideas

This section is for ideas that are not yet fully formed, but may be worth exploring in the future.

- Temperature variability: we calculate GDH variability.
  Temperature variability may be a good complement, but not implemented yet.
- MOSMIX offers solar radiation power directly as Rad1h, but not through Bright Sky API.
  We run estimation from sunshine duration, but use Rad1h in the future.
- MOSMIX offers additional variability information, like Tmax, Tmin, T5cm.
  Could be used in future for better prediction.