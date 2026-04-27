# Yield prediction

The goal of this project is to predict the yield of strawberry plants in the greenhouse for the next 28 days.

For now the contents of this project will only include everything needed for a first working version and might not take into account all the data that could be used. To see the more general ideas and all the data potentially available see: https://v-greens.atlassian.net/wiki/spaces/sai/pages/614957057/Yield+Prediction

## Input Data

There will be a training phase to collect data for 8 weeks.

Fixed input data:

- Growth degree hours (GDH) needed by the plant
    - Amount of hours the plant needs to develop a flower into a ripe fruit at a fixed temperature
    - Temperature dependent, higher the temperature means more collected GDH per actual hour
    - If we know the real temperature we can predict how many GDH each plant gets
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

### GDH needed per ripeness state

How many GDH does each state take to be classified as the next one. No tracking possible, image setup might change slightly or fruits move due to harvesting.
First guess: each state takes 1/number_states of total GDH. Use statistic model to get this values from bin counts over time and the actual harvested fruits.

### Extrapolate observed plants

Extrapolates the bins for the whole greenhouse based on the bins of the observed images. Could be a simple factor like x plants are observed and there are y plants in the facility.

### GDH prediction

Predicts the amount of GDH based on weather forecast and actual temperature data.

### Average size of fruit

We need to multiply the number of predicted ripe fruits by their average size to estimate the yield in weight. This values might change of the growing period.

Future plans: Use bounding box size to get a better estimation of fruit sizes

## Structure

- `fruitloadprediction` --> Uses current images to deliver estimated bins of the growing stages for the whole greenhouse
- `gdhprediction` --> Uses weather data to predict the amount of GDH over the next 28 days
- `yieldprediction` --> Uses current fruit load and GDH prediction to estimate the yield in weight for the next 28 days


## Model evaluation

The different models will need to be evaluated somehow. However this is difficult due to only having one actually measures value, the actual yield and many parameters that can be fitted. For this reason all parameters should have constrains set by an expert to make sure they can not get too high or too low.

### GDH prediction

For the GDH prediction we can use the actual temperature measured in the greenhouse to validate the model. This means that the GDH prediction is independent from the actual yield since it can be validated using ground truth temperature values. Therefore in the final yield prediction we will not see the GDH as a parameter but take it as a correct prediction.

### Fruit load prediction

Parameters:
- visible_plants_per_camera
    - Each camera sees a certain number of plants which we need to know for extrapolating the counts for the total greenhouse
    - Set by the user based on evaluation of images
- total_plants
    - All plants in the greenhouse, needed to extrapolate the counts for the total greenhouse
    - Set by the user
- count_correction_factor
    - This factor is used to correct the counts from the images per stage
    - If the plants are full with fruits the object detection algorithm likely misses some
    - A small flower is likely more easy to miss than a ripe fruit, hence the factor per stage
    - Ground truth values could be determined by manually counting on some images in the training phase or in the greenhouse
    - A model to correct those values based on ground truth values could be used here
    - This would reduce the number of parameters depending on the actual yield value as ground truth

## Yield prediction

Parameters:
- gdh_percent_stage_to_stage
    - Each stage needs a certain percentage of the total needed GDH to grow to the next one
    - As a ground truth we can use the counts of based on the images and the prior predictions of the model
    - We would see how counts were predicted and how many are measured such that the parameter can be corrected
- avg_weight
    - Average weight of a ripe fruit