# Greenhouse Temperature & Yield Corrections for GDH-Based Prediction

## Scope and design constraints

`gdhprediction` predicts strawberry maturation over a 28–30 day horizon by accumulating
**Growth Degree Hours (GDH)**, $\max(T - T_{base}, 0)$ summed hour-by-hour. The horizon is
built by stitching two data legs:

- a **weather leg** — a ~10-day hourly BrightSky/DWD forecast, and
- a **climate leg** — NASA POWER hourly climatology (per-hour percentiles across years).

This document describes how to correct *outdoor* forecast conditions toward the *effective
developmental conditions inside the greenhouse*. Three constraints shape every choice below.

**1. Corrections must apply to both legs, or the seam breaks.** The stitch anchors the
climate median to the weather leg's final GDH value. If we correct one leg's effective
temperature but not the other, the *accumulation rate* changes discontinuously at the
seam and biases the 28-day total. Corrections must therefore use only variables both legs
carry:

| Quantity | Weather leg (BrightSky) | Climate leg (NASA POWER) |
|---|---|---|
| Air temperature | `temperature` | `T2M` |
| Relative humidity | `relative_humidity` (or from `dew_point`) | `RH2M` (present in dtype, currently unused) |
| Shortwave irradiance | derived from `sunshine` (Ångström–Prescott) | `ALLSKY_SFC_SW_DWN` (W/m²) |
| **Wind speed / pressure** | **not fetched** | not used (`WS2M` available upstream) |

$T$, $RH$ and shortwave radiation are the safe common set. Wind and pressure are *not* symmetric across
the two legs today, which is the practical reason the wind-driven energy-balance terms are
future work rather than a near-term step.

**2. Separate the developmental clock from yield/quality.** GDH is a proxy for
*developmental rate* (phenology), which the literature treats as essentially
temperature-driven ([Wikipedia: Growing degree-day][gdd]). Humidity and heat *stress*
primarily degrade *yield, fruit set, size and quality* — not the rate of the
developmental clock. Folding those effects into $\max(T - T_{base}, 0)$ conflates two
distinct processes. We keep the clock temperature-only (Part B) and put stress effects in
a separate, downstream modifier (Part C).

**3. Calibration may not be available initially.** Indoor sensor logs are not guaranteed at
first deployment, so the default model must be runnable from *tabulated physical
parameters alone*, and must degrade gracefully into a calibrated model later **without
changing its structure**. Part A is built around this.

The work splits cleanly into three layers:

- **Part A** — estimate in-canopy air temperature from the outdoor forecast (the
  `TemperatureTransform` layer in `transforms.py`).
- **Part B** — convert effective temperature to GDH (the thermal-time *clock*).
- **Part C** — yield/quality stress modifiers (separate, future, lower-confidence).

---

## Part A — Outdoor → in-canopy temperature

### A single linear model with two parameterisation routes

Treat the greenhouse as a **lumped, well-mixed air volume** — the standard simplification for
commercial houses, and one that is *valid only when the air is actually mixed*. Internal
horizontal-airflow (HAF) circulation fans are what make this true: they suppress vertical
stratification so that a single $T_{in}$ is meaningful. They add no term to the balance — they
just make the existing one honest. The steady-state sensible-heat balance over the air volume
is

$$
\begin{aligned}
\underbrace{\tau A_f G}_{Q_{solar}} + Q_{heat}
&= \underbrace{U A_{cover}(T_{in}-T_{out})}_{Q_{cond}} \\
&\phantom{=}+ \underbrace{\dot m_{vent} c_p (T_{in}-T_{out})}_{Q_{vent}} \\
&\phantom{=}+ \underbrace{\rho c_p \tfrac{V_{gh} N_{inf}}{3600}(T_{in}-T_{out})}_{Q_{infil}} \\
&\phantom{=}+ Q_{latent}
\end{aligned}
$$

On the input side, $Q_{solar}$ is the transmitted solar gain and $Q_{heat}$ the sensible heat
supplied by the heating system, if any (W); the right-hand side collects the losses. The
remaining symbols are: $\tau$ glazing transmissivity; $G$ global irradiance on a horizontal
surface (W/m²); $A_f$ the floor footprint area — the area that intercepts the horizontal
irradiance, and *distinct from* the larger cover area $A_{cover}$ that governs conduction loss;
$U$ the cover heat-transfer coefficient (W/m²·K); $c_p \approx 1006$ J/kg·K the specific heat
capacity of air; $\rho \approx 1.2$ kg/m³ the density of air; and $V_{gh}$ the house air volume.

Two distinct air-exchange terms appear, and the difference matters:

- $\dot m_{vent}$ (kg/s) is the **deliberate ventilation** mass flow — outside air driven
  through the house by exhaust fans or open vents under operator/control command. It is set by
  the grower and can be switched on, off or staged.
- $N_{inf}$ is the **uncontrolled infiltration** rate (air changes per hour) — leakage through
  gaps, panel laps and door seals that happens regardless of whether the vents are open. The
  term $\rho c_p V_{gh} N_{inf}/3600$ converts it into a heat-loss coefficient (W/K).

Finally, $Q_{latent}$ is the latent heat carried off by crop transpiration: water evaporating
from the leaves absorbs energy and removes it from the air as water vapour rather than as a
temperature change, so it is an energy sink that never appears as sensible heat.

Rather than carry $Q_{latent}$ as a fixed constant, we adopt the ASAE/Mihara closure used in the
greenhouse-design literature ([Al-Helal et al. 2015][ventmodel]): the transpiration sink scales
with the transmitted solar gain, $Q_{latent} = \delta\,\tau A_f G$, where $\delta$ is the
**evaporation coefficient** — the fraction of transmitted solar consumed by transpiration. The
transmitted-minus-latent input then folds into a single **solar heating factor**

$$\eta = (1-\delta)\,\tau,$$

so the net radiative gain is $\eta A_f G$. Solving the balance for indoor temperature:

$$
T_{in} = T_{out} + \underbrace{\frac{Q_{heat}}{K}}_{a}
+ \underbrace{\frac{\eta A_f}{K}}_{b}\, G,
$$

where the **heat-loss denominator** $K$ (W/K) collects the conduction, ventilation and
infiltration losses:

$$K = U A_{cover} + \rho c_p \dot V_{vent} + \rho c_p \frac{V_{gh} N_{inf}}{3600}.$$

The crucial structural fact: **whenever $K$ is approximately constant, this collapses to**

$$\boxed{\,T_{in} = T_{out} + a + b\,G\,}$$

which is exactly the existing `RadiationShift` transform (`degree_shift` $\leftrightarrow$ $a$, `solar_factor` $\leftrightarrow$
$b$; see `transforms.py` — the config field `temperature_shift` is supplied as the transform's
`degree_shift`), with $a = Q_{heat}/K$ and $b = \eta A_f / K$. The Al-Helal loss denominator
$\dot m_a C_{pa} + U A_c$ is identical to $K$ up to the infiltration term; their tabulated values are
$\eta \approx 0.3$–$0.7$ and $\delta \approx 0.2$–$0.5$ (independent ranges — at $\delta \approx 0.5$,
$\eta = (1-\delta)\tau$ caps near $0.5$).

So the same two coefficients can be obtained either way:

- **Route A-i — forward physics (no calibration).** Look up $\eta, U, A_f, A_{cover}, V_{gh},
  N_{inf}$ from greenhouse geometry and standard tables, read the fan flow $\dot V$ off the fan
  nameplate, add any heating input $Q_{heat}$, and *compute* $a$ and $b$. Runnable on day one.
- **Route A-ii — calibrated regression (later).** Once indoor temperature is logged, fit
  $a, b$ by linear regression of $(T_{in} - T_{out})$ on $G$. Same model, better coefficients.

You never switch models — only re-source two numbers: start with physics-derived
coefficients, refit them when sensor data arrives.

Route A-i needs a value for the evaporation coefficient $\delta$ (equivalently $\eta$). We adopt the
tabulated empirical $\delta \approx 0.5$ as a convenient simplification: holding $\delta$ constant
keeps $b = \eta A_f / K$ constant in $G$, preserving the $T_{in} = T_{out} + a + bG$ linearity the
GDH clock relies on, and $\delta$ is most stable precisely in the well-watered, high-LAI regime
where that matters most. A transpiration-derived $\delta$ (Stanghellini-type) is a documented future
refinement, not a near-term step — see Appendix A.

### The ventilation regime sets $b$ — one model, sourced per house

We target a range of houses: **computer-managed** (climate-computer setpoint control),
**manually operated** (a grower opening vents by hand), and **passive** (fixed or roll-up
apertures left to the weather). The single linear form covers all three — they differ only in
*how $a, b$ are sourced and which level (A0–A2) is appropriate*, not in the equation. The lever
is the ventilation term in $K$, and it spans an order of magnitude. Worked example — a ~1000 m²
floor house ($A_f = 1000$ m², cover area $A_{cover} \approx 1400$ m², air volume
$V_{gh} \approx 3000$ m³ at ~3 m mean height), $\eta = 0.5$, $U = 4$ W/m²·K:

- **Heavily mechanically ventilated** at ~1 air-volume per minute ($\dot V \approx 50$ m³/s):
  $\dot m_{vent} c_p \approx 1.2 \cdot 50 \cdot 1006 \approx 60{,}000$ W/K dominates
  $U A_{cover} \approx 5{,}600$ W/K (~91 % of $K$), giving
  $b = \eta A_f / K \approx 0.5 \cdot 1000 / 65{,}600 \approx 0.008$ °C per W/m².
- **Closed / infiltration only** ($N_{inf} \approx 1$): the infiltration loss
  $\rho c_p V_{gh} N_{inf}/3600 \approx 1{,}000$ W/K gives $K \approx 6{,}600$, so
  $b \approx 0.08$ °C per W/m² — a ~10× larger radiation swing.

The configured default `solar_factor` $\approx 0.01$ therefore corresponds to the *fans-on* end
of that range, **not** a universal value. The practical conclusion: **$b$ must be sourced
per-house** — from physics (Route A-i) for a known geometry/aperture, or from calibration
(Route A-ii) once sensors exist — rather than assumed. That per-house sourcing is exactly what
lets one structure serve managed, manual and passive houses alike.

Mechanical ventilation is the regime where the *forward* route is cleanest: a nameplate fan flow
$\dot V$ makes $\dot m_{vent} = \rho \dot V$ known and constant, so $K$ is constant by
construction. Natural ventilation (vents, roll-up sides — the dominant case in Central-European
strawberry tunnels) has no fan term: $\dot V$ is set by aperture, wind and the $\Delta T$ stack
effect. We treat aperture-within-a-setting as fixed and **deliberately omit wind to keep the
model simple** (it is also not fetched symmetrically across the two legs); the residual
load-coupling of $K$ is handled as a documented limitation below, not a structural term. This is
consistent with the greenhouse-modelling literature, where naturally and mechanically ventilated
houses alike are routinely modelled with linear regressions on outdoor temperature and solar
radiation ([linear/ARMAX and semi-empirical balance models][ventmodel]; ventilation design flows
in [UF/IFAS *Greenhouse Ventilation*][ifas]).

### The levels mapped to `transforms.py`

| Level | Form | `transform_key` | Parameters |
|:---|:-------|:-------|:----------|
| A0 | $T_{in} = T_{out}$ | `identity` | none |
| A1 | $T_{in} = T_{out} + a$ | `shift` | `temperature_shift` ($a$) |
| A2 | $T_{in} = T_{out} + a + bG$ | `radiation_shift` | `temperature_shift` ($a$), `solar_factor` ($b$) |

- **A0 `Identity`** — outdoor baseline; useful as a control and for open-field comparison.
- **A1 `Shift`** — mean passive warming. Captures the daily-mean offset but misses the
  radiation-driven diurnal swing, so it under-/over-shoots on clear days.
- **A2 `RadiationShift`** — the reduced-form energy balance above. **Recommended default**,
  because (i) it captures the radiation-driven swing, (ii) under ventilation its coefficients
  are physically computable without calibration (Route A-i), and (iii) it later absorbs
  calibration (Route A-ii) without a structural change.

### Caveats and the future step

- **Steady-state ignores thermal mass** (no $C\,dT/dt$), so the model overshoots diurnal
  extremes; since GDH rectifies with $\max(\cdot,0)$, daytime overshoot inflates accumulation —
  the main systematic error. High air exchange reduces it (it weakens air–structure coupling), so
  it is mildest in well-ventilated regimes.
- **Natural-ventilation load-coupling — the dominant case.** In passively or manually vented
  houses (vents, roll-up sides) $K$ has no fan term: $\dot V$ rises as the aperture opens with
  cooling demand and with outdoor wind, so the *true* $b$ falls as radiation/temperature rise and
  saturates once vents are fully open. By treating aperture-within-a-setting as fixed and omitting
  wind, we fit a single load-averaged $b$; the residual error is concentrated at high-radiation
  midday — unfortunately where the $b\,G$ term is largest and GDH accrues fastest. Constant $K$ is
  most defensible where the aperture is genuinely fixed (passive houses, between a grower's manual
  adjustments) and at night / low radiation when vents sit at minimum; it is weakest under a
  proportional climate computer that modulates vents continuously — but see the setpoint caveat,
  which takes over precisely there.
- **Staged fan control — the minor case.** Where mechanical exhaust fans exist they typically run
  staged on/off, making $\dot V$ a step function of cooling demand and $K$ piecewise-constant; the
  linear form holds per stage, with mild nonlinearity across stage transitions. This is the
  exception rather than the rule for the target market.
- **HAF circulation fans do not enter $K$.** Horizontal-airflow fans only mix internal air; they
  do not exchange air across the cover, so they contribute nothing to $\dot V_{vent}$ — they make
  the well-mixed single-$T_{in}$ assumption honest (Part A intro) without changing the balance.
- **Active setpoint control overrides physics.** If the house is thermostatted (the
  computer-managed tier), the setpoint schedule is a stronger prior than any forward model during
  controlled hours; the energy balance then governs only the uncontrolled excursions. This is also
  what rescues the constant-$K$ idealisation in the one regime (proportional vent control) where it
  is otherwise weakest.
- **A3 (future) — wind/ventilation energy balance.** Wind is the *mechanism* that sets $K$ in the
  dominant natural-ventilation regime, consciously deferred here for simplicity rather than merely
  a nice-to-have. Extending A2 with wind-driven convection and natural-ventilation terms is the
  calibration endpoint, gated on (i) adding wind to *both* fetches (BrightSky `wind_speed`, NASA
  `WS2M`) to keep the legs symmetric, and (ii) indoor sensors to fit against. Until both exist, it
  adds tabulated parameters without a reliable accuracy gain.

---

## Part B — Effective temperature → GDH, the thermal clock

GDH accumulates $\max(T_{in} - T_{base}, 0)$ per hour. Developmental rate from emergence to maturity
depends on air temperature, which is what makes thermal-time accumulation predictive
([Wikipedia: Growing degree-day][gdd]).

### Base temperature

The configured `gdh_base` ($4.5$ °C) is well-supported: strawberry phenology is commonly modelled
with a base near 3 °C (e.g. flower→fruit thermal-time models, [MDPI *Agronomy* 2023][mdpi]),
within a broader literature range of ~0–7 °C. No change recommended.

### Keep the clock linear above base — do **not** cap at the optimum

The generic agronomy convention caps GDD above ~30 °C ([Wikipedia: Growing degree-day][gdd]),
but the strawberry-specific evidence argues against capping the *clock*: above the optimum,
flower and fruit development *accelerate* and days-to-ripening *shorten* (faster at 30/25 °C
than at 23/18 °C) ([review][review]; [heat-stress study][heat]). High temperature instead
degrades *fruit set, size, colour and quality* (~70–80 % fruit-weight loss at 30 °C) — i.e.
yield, not timing. That penalty belongs in Part C.

### Optional high heat-stress cap — numerical guard only

A cap is defensible *only* near a true heat-stress / fruit-abortion threshold (~34–35 °C), and
then only as a **guard against `RadiationShift`-inflated midday extremes**, not as a
phenological statement. Such temperatures are rare under DWD coverage, so this is a minor,
optional refinement.

### Rejected clock modifiers

- **Wet-bulb temperature as the developmental temperature.** Wet-bulb represents an
  *evaporating wet* surface (a water-wicked thermometer sock; [Wikipedia: Wet-bulb
  temperature][wetbulb]); no source equates dry fruit-pulp temperature to wet-bulb. Its
  legitimate orchard/disease-model uses concern *wetted* leaf/dew surfaces. Strawberry fruit
  transpire weakly and are radiatively heated, running *above* air temperature in sun — so
  using wet-bulb would bias the clock cold during exactly the warm daytime hours that dominate
  GDH. Rejected as a clock input.
- **VPD-scaling of GDH.** High vapour-pressure deficit closes stomata and reduces CO$_2$
  assimilation ([Grossiord et al. 2020][grossiord]; [Frontiers Plant Sci. 2021][frontiers]) —
  a *carbon-gain / growth* effect, not a change in developmental rate. Scaling the clock by a
  VPD modifier conflates growth limitation with phenological rate. Moved to Part C.

---

## Part C — Yield & quality stress modifiers

These do **not** enter the GDH clock. They are downstream multipliers/penalties on predicted
*yield or fruit quality*, applied after maturation timing is established. Both required inputs
($T$, $RH$) are already present in both legs, so no new data source is needed — but each needs
local validation before production use.

- **High-temperature yield/quality penalty.** Above ~25–30 °C, fruit set, individual fruit
  weight, size and colour decline sharply ([review][review]; [heat-stress study][heat]). A
  per-stage penalty function on yield is the appropriate home for this effect.
- **High-VPD assimilation penalty.** Sustained high VPD reduces stomatal conductance and
  assimilation ([Grossiord et al. 2020][grossiord]; [Frontiers Plant Sci. 2021][frontiers]).
  VPD is derived from $T$ and $RH$ (or `dew_point`) via the Magnus relation already used in
  `utils.rh_from`. A linear ramp between a crop-specific $VPD_{opt}$ (~0.8–1.2 kPa) and
  $VPD_{max}$ (~2.0 kPa) is a reasonable first form.

Status: **future work**, explicitly flagged as lower-confidence and calibration-dependent.

---

## Data pipeline mapping

All required *temperature-clock* inputs are available today from the existing fetches:

| Need | Source today | Notes |
|:--------|:--------------------|:--------------------|
| $T_{out}$ (both legs) | BrightSky `temperature`; NASA `T2M` | direct |
| Shortwave $G$ (both legs) | BrightSky: derived from `sunshine` via Ångström–Prescott (`sunshine_to_radiation`); NASA `ALLSKY_SFC_SW_DWN` | BrightSky value is *modelled*, not measured (Ångström–Prescott on extraterrestrial radiation, $R_a(a_s + b_s n)$) — a known uncertainty that propagates into $Q_{solar}$ |
| $RH$ (Part C) | BrightSky `relative_humidity` or `dew_point`; NASA `RH2M` | NASA `RH2M` is in the dtype but currently unused |
| Wind (A3 only) | **not fetched** | BrightSky `wind_speed`; NASA `WS2M` — both needed to keep legs symmetric |
| Geometry/thermal params (A-i) | greenhouse spec + standard tables | $\eta$, $U$, $A_f$, $A_{cover}$, $V_{gh}$, $N_{inf}$, fan flow $\dot V$, heating |

---

## Summary of recommendations

| | Layer | Recommendation | Calibration |
|---|---|---|---|
| **A2** | In-canopy temperature | **Default**: `RadiationShift` as reduced-form energy balance; coefficients from tabulated physics now (Route A-i), refit from sensors later (Route A-ii) | Not required to start |
| A1 / A0 | In-canopy temperature | `Shift` / `Identity` as fallbacks and controls | None |
| A3 | In-canopy temperature | Wind/ventilation extension — **future**, gated on symmetric wind data + indoor sensors | Required |
| **B** | Thermal clock | Keep $\max(T - 4.5, 0)$ **linear above base**; optional high heat-stress cap (~34–35 °C) as a numerical guard only | None |
| **C** | Yield/quality | High-temperature and high-VPD penalties **downstream of GDH** — future, low-confidence | Required |

The throughline: a single linear in-canopy-temperature model that is physically derivable
without calibration and calibration-ready later; a deliberately simple, temperature-only
thermal clock; and a clean separation of all stress effects into a downstream yield/quality
layer.

---

## Appendix A — Sourcing the evaporation coefficient $\delta$

Route A-i needs a value for the evaporation coefficient $\delta$ (equivalently $\eta$). The default
is the tabulated empirical $\delta \approx 0.5$. A more principled alternative *is* available — and
worth recording even though we do not adopt it by default. By definition $\delta$ is the share of
transmitted solar consumed by transpiration, $\delta = \lambda E / (\tau A_f G)$, where
$\lambda \approx 2.45$ MJ/kg is the latent heat of vaporisation and $E$ the canopy transpiration
mass flux. This is the standard *evaporative fraction* ($EF = \lambda E/(R_n - G)$, the complement
of the Bowen ratio), normalised here to transmitted shortwave $\tau G$ rather than net radiation —
so it is close to, but not identical with, a textbook $EF$. Given an estimate of $E$, $\delta$
follows. The canonical greenhouse transpiration model is **Stanghellini (1987)**
([thesis][stanghellini]) — a Penman–Monteith big-leaf equation with a $2\,LAI$ canopy multiplier
and a greenhouse-specific (low-wind) aerodynamic resistance. Conveniently for us, under forced HAF
circulation the leaf aerodynamic resistance $r_a$ is set by the roughly-constant fan-driven
in-canopy air speed and the free/mixed-convection regime, **not** by outdoor wind — so this route
would *not* require the wind data the two legs do not carry symmetrically (this fails for naturally
ventilated houses, where vent flow couples to outdoor wind).

We keep the empirical $\delta$ as the default for two reasons. First, a transpiration estimate
re-imports exactly the parameters the lumped $\delta$ hides: leaf-area index $LAI$ (time-varying,
needs a phenology proxy), canopy/stomatal resistance $r_c$ (a Jarvis-type response in radiation,
VPD and temperature), and net radiation $R_n$ (longwave/absorptivity assumptions on top of our
transmitted shortwave). Second, a radiation- and VPD-dependent $\delta$ makes $b = \eta A_f/K$ no
longer constant in $G$, perturbing the $T_{in} = T_{out} + a + bG$ linearity the GDH clock relies
on. $\delta$ is most stable precisely in the well-watered, high-LAI regime — which is where holding
it constant is most defensible anyway. Reported values: well-watered greenhouse tomato shows
$EF \approx 0.66$–$0.72$ (Bowen ratio $0.25$–$0.32$), and the Al-Helal/Abdel-Ghany arid-greenhouse
work puts the latent-heat factor at $\sim0.22$–$0.5$ and emphasises it is not a constant
([Abdel-Ghany & Al-Helal 2011][abdelghany]). No strawberry-specific $EF$ appears in the literature;
strawberry's lower stature and LAI suggest a somewhat lower $\delta$ than tomato, but that is an
extrapolation. A transpiration-derived $\delta$ is therefore a documented future refinement (gated
on $LAI$ + canopy-resistance inputs), not a near-term step.

---

## References

- [Stull (2011), *Wet-Bulb Temperature from Relative Humidity and Air Temperature*, J. Appl. Meteorol. Climatol. 50(11)][stull] — wet-bulb approximation; errors −1 °C to +0.65 °C, mean abs < 0.3 °C, valid RH 5–99 %, T −20–50 °C.
- [Allen et al. (1998), *FAO Irrigation & Drainage Paper 56*][fao56] — Ångström–Prescott / sunshine-to-radiation, used in `utils.sunshine_to_radiation`.
- [Wikipedia: Growing degree-day][gdd] — thermal-time/developmental-rate basis; base and upper-cutoff conventions.
- [Wikipedia: Wet-bulb temperature][wetbulb] — physical meaning of wet-bulb (evaporating wet surface).
- [MDPI *Agronomy* 13(10):2489 (2023), strawberry phenological-cycle model][mdpi] — strawberry thermal-time / base temperature.
- [Review: high temperatures accelerate strawberry flower development and decrease fruit size (2023)][review] — high temperature accelerates development, reduces size/quality.
- [Effect of high-temperature stress on reproductive growth of strawberry, *Sci. Hortic.*][heat] — fruit-set/size loss and shortened days-to-ripening at high temperature.
- [Grossiord et al. (2020), *Plant responses to rising VPD*, New Phytologist][grossiord] — VPD → stomatal closure → reduced assimilation.
- [Minimizing VPD fluctuations maintains stomatal conductance and photosynthesis, *Front. Plant Sci.* (2021)][frontiers] — VPD effect on conductance/assimilation.
- [Modified thermal model to predict natural ventilation of greenhouses, *Energy & Buildings* (2015)][ventmodel] — linear/semi-empirical greenhouse temperature models.
- [UF/IFAS AE-10, *Greenhouse Ventilation*][ifas] — mechanical ventilation design flow rates.
- [Stanghellini (1987), *Transpiration of greenhouse crops: an aid to climate management*, PhD thesis, Wageningen][stanghellini] — canonical greenhouse Penman–Monteith transpiration model ($2\,LAI$ multiplier, low-wind aerodynamic resistance).
- [Abdel-Ghany & Al-Helal (2011), *Energy partition and conversion of solar and thermal radiation…*, Energy & Buildings 43(7):1740–1747][abdelghany] — sensible/latent heat factors and evaporative fraction in arid greenhouses; factors are not constant.

[stull]: https://journals.ametsoc.org/view/journals/apme/50/11/jamc-d-11-0143.1.xml
[fao56]: https://www.fao.org/4/x0490e/x0490e00.htm
[gdd]: https://en.wikipedia.org/wiki/Growing_degree-day
[wetbulb]: https://en.wikipedia.org/wiki/Wet-bulb_temperature
[mdpi]: https://www.mdpi.com/2073-4395/13/10/2489
[review]: https://www.researchgate.net/publication/368289728_A_review_of_fruit_development_in_strawberry_high_temperatures_accelerate_flower_development_and_decrease_the_size_of_the_flowers_and_fruit
[heat]: https://www.sciencedirect.com/science/article/abs/pii/S0304423807004256
[grossiord]: https://nph.onlinelibrary.wiley.com/doi/10.1111/nph.16485
[frontiers]: https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2021.646144/full
[ventmodel]: https://www.sciencedirect.com/science/article/abs/pii/S0378778815003126
[ifas]: https://ask.ifas.ufl.edu/publication/AE030
[stanghellini]: https://edepot.wur.nl/202121
[abdelghany]: https://www.sciencedirect.com/science/article/abs/pii/S0378778811000958