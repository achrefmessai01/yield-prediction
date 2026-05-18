# Yield Prediction

## About

This small project should allow simple yield prediction based on a few parameters.
It is meant for a greenhouse of unspecific geometry and size.
The greenhouse has several cameras installed, which are able to detect the fruits and their maturation stages.
An ML program classifies the fruits into _seven_ maturation stages.
To graduate from one stage to the next, the fruit needs to be exposed to certain temperatures.
Based on the observed and classified fruit sample, we forecast the maturation schedule.
We extrapolate the yield for the whole greenhouse.

## Details

The temperature is the main driver of strawberry maturation.
We are able to sample about 40 plants using 10 cameras.
Each fruit maturation stage requires a certain amount of growth-degree-hours (GDH).

The program must meet the following:

- [ ] take counts of seven strawberry maturation stages as input
- [x] read historical temperature data from NASA Power API
- [x] use local meteorological forecast (e.g. BrightSky)
- [ ] predict the maturation schedule for the next 30 days
- [ ] extrapolate the yield for the whole greenhouse
- [x] for a given GPS, download the historical temperature data only once

Configuration must allow:

- [ ] different greenhouse setups: active, passive, offset temperature
- [ ] configuration of the number of cameras and their sampling weights
- [ ] number of plants in camera view, number of plants in total

Stages modelling:

- [ ] each stage obtained from the classifier should be mapped onto a range of GDH
- [ ] use normal (or other?) distribution for the mapping
- [ ] the GDH range should be configurable for each stage

Strawberry classification:

- [ ] use classifier by Fabian in `ml-platform` repository

## Technical details

The program will run on server.
The program will have access to site sensor data from a database.
The accessible data are temperature (T2M, °C), relative humidity (RH2M, %), CO2 (CO2, ppm),
and radiation measurements (lux).
