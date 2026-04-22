# Yield prediction

The goal of this project is to predict the yield of strawberry plants in the greenhouse for the next 28 days.

For now the contents of this project will only include everything needed for a first working version and might not take into account all the data that could be used. To see the more general ideas and all the data potentially available see: https://v-greens.atlassian.net/wiki/spaces/sai/pages/614957057/Yield+Prediction

## Input Data

There will be a training phase to collect data for 8 weeks.

Fixed input data:

- Growing hours needed by the plant
    - Amount of hours the plant needs to develop a flower into a ripe fruit at a fixed temperature
    - Temperature dependent, higher the temperature means more collected growing hours per actual hour
    - If we know the real temperature we can predict how many growing hours each plant gets
    - The grower knows this values for each plant
- Number cameras, amount of plants each camera monitors, amount of plants in total

Following data will be collected and used by the models:

| Data          | Format        | Frequency     |
| ------------- | ------------- | ------------- |
| Weather data  | ?             | ?             |
| Greenhouse temperature | °C   | Every 5 min   |
| Stage counts  | bin of the 6 stages | Twice a day | 
| Actual yield  | weight and counts | ?         |

## Sub-Tasks

For a ordered overview of the sub tasks see Jira: https://v-greens.atlassian.net/browse/DS-2?atlOrigin=eyJpIjoiZDdjNDliYTg4NTM2NGJlMGI5ZTZmYjcyOWVjNzE0MjUiLCJwIjoiaiJ9

### Image object detection

Edgedevices with cameras will take images periodically, evaluate them using a model and send back bins (counts of 6 classes detected in the image). Each class is a ripeness state from flower to ripe fruit.

### Growing hours needed per ripeness state

How many growing hours does each state take to be classified as the next one. No tracking possible, image setup might change slightly or fruits move due to harvesting.
First guess: each state takes 1/number_states of total growing hours. Use statistic model to get this values from bin counts over time and the actual harvested fruits.

### Extrapolate observed plants

Extrapolates the bins for the whole greenhouse based on the bins of the observed images. Could be a simple factor like x plants are observed and there are y plants in the facility.

### Growing hour prediction

Predicts the amount of growing hours based on weather forecast and actual temperature data.

### Average size of fruit

We need to multiply the number of predicted ripe fruits by their average size to estimate the yield in weight. This values might change of the growing period.

Future plans: Use bounding box size to get a better estimation of fruit sizes

## Structure

- `fruit_load_model` --> Uses current images to deliver estimated bins of the growing stages for the whole greenhouse
- `growing_hour_model` --> Uses weather data to predict the amount of growing hours over the next 28 days
- `yield_prediction_model` --> Uses current fruit load and growing hour prediction to estimate the yield in weight for the next 28 days