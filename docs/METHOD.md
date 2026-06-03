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

## Time frame and daylight saving time

The two legs live in different hourly frames; the splice has to reconcile them.

- **Climatology (NASA POWER, `time-standard=lst`)** indexes hours in *local standard time* — the wall-clock offset of the site's IANA zone with DST switched off — and uses a fixed 365×24 layout. Climatology has no concept of DST transitions; June 15 12:00 and December 15 12:00 are simply hour-of-year offsets in the same standard-time clock.
- **Weather (BrightSky)** is requested with an explicit UTC offset on the `date` ISO string. The BrightSky `/weather` contract is: when `tz=` is not supplied, all returned records are stamped with the offset taken from `date`. We send the offset that the IANA zone has at `start_datetime`. There is exactly one record per UTC hour for the full request span — DST transitions inside the window do *not* produce 23-hour or 25-hour days in the response.

Consequently the weather frame coincides with the climatology frame whenever `start_datetime` is in *standard* time (the request offset equals LST). When `start_datetime` is in *DST*, the weather frame runs one civil hour ahead of climatology for the whole response — including across any transition that falls inside the 10-day window. The seam alignment therefore depends only on whether `start_datetime.dst()` is non-zero:

| `start_datetime` in DST? | Climate index at seam |
| --- | --- |
| No (standard time) | `len(weather) + datetime_hour` |
| Yes (summer time)  | `len(weather) + datetime_hour − 1` |

The −1 pulls the climate continuation back by one hour-of-year cell so the post-seam GDH curve resumes from the correct LST hour and stays smooth across the join. See the *Limitations* section for the consequences for downstream array consumers and for the assumptions baked into this rule.

## Aggregation

The native output is hourly CUM_GDH (median + percentile envelope). Daily output is obtained by sampling the last hour of each local day.

## From CUM_GDH to harvest (planned)

Cameras classify a sample of fruits into seven maturation stages. Each stage corresponds to a distribution over remaining CUM_GDH-to-harvest (default: normal, configurable). Sample counts are extrapolated to whole-greenhouse counts by the camera sampling weights. Convolving each stage's residual-GDH distribution with the forecast CUM_GDH(t) yields an expected harvest-rate curve over the horizon, with uncertainty propagated from the climatology percentile band.

## Limitations

### Output array is not a civil-clock series

The returned hourly array is **uniformly spaced in UTC** (one entry per UTC hour). Its *wall-clock* meaning differs between the two legs:

- Indices into the weather leg correspond to the offset present at `start_datetime` — i.e. CEST during summer requests, CET during winter requests. There is no offset change *within* the weather leg, even if a DST transition falls in the 10-day window: BrightSky keeps reporting in the start-instant's offset, so e.g. a "02:00" entry on a spring-forward day is actually 03:00 civil time.
- Indices into the climate leg correspond to local *standard* time hours (LST = the zone's winter offset), regardless of the request's season.

Practical consequences for downstream consumers:

- The array is **not** a civil-clock wall-time series. Plotting array index against "local time" without a per-entry timestamp will misplace samples by one hour during summer-time requests and at the weather→climate seam when the request straddles seasons.
- The daily aggregation samples every 24 entries (last hour of each 24-hour block). Because the underlying axis is fixed-offset, each daily sample corresponds to "the 23rd hour of the standard-time day" — not necessarily civil 23:00. In practice this matters when downstream code reads the daily series against a wall-clock calendar.

Callers that need civil-clock alignment should either resample against an external civil timestamp axis, or request the prediction with `start_datetime` deliberately set to a standard-time moment so weather and climate share the LST frame end-to-end.

### DST handling is coupled to BrightSky/DWD

The seam-alignment rule (see *Time frame and daylight saving time*) hard-codes two assumptions that hold for the current weather provider only:

1. The weather provider returns records in the fixed UTC offset of `start_datetime` (BrightSky's behaviour when no `tz=` parameter is supplied).
2. Climatology is in local *standard* time (NASA POWER `time-standard=lst`).

Both are true today because BrightSky is DWD-bound (Germany / central Europe). Swapping in a non-DWD provider for global coverage would invalidate the rule in ways that depend on the new provider's contract:

- A provider that returns DST-aware civil-time timestamps (e.g. Open-Meteo with a `timezone` parameter) makes the seam correction depend on what falls *inside* the window — specifically whether the last weather hour is in DST — rather than on `start_datetime.dst()` alone.
- A provider that returns 23-hour or 25-hour days at DST transitions breaks the implicit `len(weather_data) == duration × 24` assumption and the `prediction[23::24]` daily-resampling stride.
- A provider that returns timestamps in UTC makes the correction a fixed per-location offset (the zone's standard-time offset) independent of the request's season.

Additionally, the snap-hour `replace(hour=…)` calls in the weather request path use `{0, 6, 12, 18}` as bucket boundaries. In DWD coverage these never coincide with a DST transition hour (Europe transitions at 02:00 / 03:00). In zones where the transition *is* at midnight — Egypt, Greenland, Chile, Lebanon at time of writing — `replace(hour=0)` would land on a non-existent or ambiguous wall-clock time. ZoneInfo silently uses `fold=0` and picks a notional pre-transition offset, so the ISO instant sent to the provider would be off by one hour from civil midnight for roughly four hours per year per site.

When adding a non-DWD provider, both points need re-derivation:

- Factor the time-frame contract out explicitly (e.g. an enum `WeatherFrame ∈ {FIXED_OFFSET_AT_START, CIVIL_TIME, UTC}` and a per-frame seam-correction helper), so the divergence between providers is localised in one place.
- Re-evaluate the snap-hour `replace(...)` calls against the new providers' transition-hour distribution; in worst case, replace them with a fold-aware helper that re-resolves the offset after rewriting the wall-clock hour.

## Deferred ideas

This section is for ideas that are not yet fully formed, but may be worth exploring in the future.

- Temperature variability: we calculate GDH variability.
  Temperature variability may be a good complement, but not implemented yet.
- MOSMIX offers solar radiation power directly as Rad1h, but not through Bright Sky API.
  We run estimation from sunshine duration, but use Rad1h in the future.
- MOSMIX offers additional variability information, like Tmax, Tmin, T5cm.
  Could be used in future for better prediction.