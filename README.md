![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<div align = "center">

<h1>Churn-prediction</h1>
</div>

Classification model to predict possible clients who want to close their bank account and define their characteristics to create a custom-made product

## Table of contents
[Description](#Description)  
[Installation](#Installation)  
[Usage](#Usage)  
[Output](#Output)  
[How it works](#How-it-works)  
[Examples](#Examples)  
[Authors](#Authors)

## Description
The API return the prediction the price of a propertie in Belgium, based on data scrapped from Immoweb. 
For the predictions our Machine Learning model looks at the relationship between the postal code, the state of the construction, the property subtype (apartment, studio, villa, chalet, ...), and existance of a fireplace, terrace, garden and/or fully equiped kitchen, an estimate of the asking price is made.

The accuracy of the model is pf  85%, which means that there is always a possibility for outliers (less then 15 %).
  
This API has been deployed with heroku under the url: https://api-ie-predictions.herokuapp.com/

## Installation

Clone the repository:
```
git clone https://github.com/jejobueno/ImmoEliza-API
```

Install the requirements
```
pip install -r requirement.txt
```

## Usage
  
For the predictions, send a `POST` request to https://api-ie-predictions.herokuapp.com/predict with the following parameters:
  
  ```json
{
  "data": {
      "area": float,
      "subpropertyType": Optional['HOUSE', 'VILLA', 'EXCEPTIONAL_PROPERTY', 'APARTMENT_BLOCK',
          'MANSION', 'MIXED_USE_BUILDING', 'BUNGALOW', 'TOWN_HOUSE',
          'FARMHOUSE', 'COUNTRY_COTTAGE', 'MANOR_HOUSE', 'APARTMENT',
          'PENTHOUSE', 'DUPLEX', 'TRIPLEX', 'LOFT', 'FLAT_STUDIO',
          'SERVICE_FLAT', 'GROUND_FLOOR'],
      "bedroomsCount": int,
      "postalCode": int,
      "ladnSurface": float,
      "hasGarden": binary bool,
      "gardenSurface": float,
      "hasFullyEquippedKitchen": binary bool,
      "hasSwimmingPool": binary bool,
      "hasFireplace": binary bool,
      "hasTerrace": bianry bool,
      "terraceSurface": float,
      "facadeCount": int,
      "buildingCondition": Optional["TO_BE_DONE_UP" , "AS_NEW" , "GOOD" , "JUST_RENOVATED" , "TO_RESTORE"]
      }
}
```

Then the result from the API will be:
  ```json
{
      "prediction" : float
}
```
If there is any error on the type of the data, formatting or fields missing. The result willl be:

  ```json
{
      "prediction" : Optional[str]
}
```
## How it works
1. Processor
First, the data are cleaned. That means that we drop all the entirely empty rows, string values
are cleaned up, outliers and properties without price and area indication are dropped, duplicates
and columns with the lowest correlation rate are deleted, and some other minor riddances.  

To put everything ready for the rest of the process, the variables that remain are transformed into
features.

2. Model
In the second step, the prediction is prepared. Firstly, the price, area, outside space and land
surface are rescaled. This is done in order to apreciate more linealy the relationship between price and area.

Secondly, the database is split and into a train and test dataframe. The first one is used to train the model.

Then we score our model, getting a 85% of accuracy in hour predictions.

3. Predictor 
This object is going to be initializated when the app.py is runned. This predictor will load the model which is already trained to make the prediction.

The data is checked to see if there is any error in the format or/and type, then preprocessed and it columns reformated in order to get a matrix with the required size and pased trough our model to get the prediction.

4. app.py
Here is where the `POST` and `GET` requests are processed. 

## Author
Jesús Bueno - Project Manager/dev & doc  
