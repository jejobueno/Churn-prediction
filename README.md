[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
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
The Churn Prediction app, contains a brief description of the data set used and the exploratory data analysis.  It also contains a form in the 'predict' section to make predictions. 

After checking the performance of three diferent model (RandomForest, ADA Boost and Support Vector Machine (SVM). It was decided to used RandomForest as main prediction model.

This APP was created with streamlit and has been deployed with heroku under the url: https://app-churn-pred.herokuapp.com/

## Installation

Clone the repository:
```
git clone https://github.com/jejobueno/Churn-prediction
```

Install the requirements
```
pip install -r requirement.txt
```

## Usage
  
As says before. This web APP can be accessed by the url https://app-churn-pred.herokuapp.com/.

It contains three seccions

## How it works
1. DataProcessor
This object is in chard to helpo with the visualization of the diferent features contained by the data. Then
this dataset is preprocessed, meaning that we drop all the entirely empty rows, string values
are cleaned up, and features with the higher correlation between each other are removed, mantaining the one
with the highest spearman correlation in relation to the target feature.

2. Predictor 
This object is going to be load when the app.py is runned. This predictor will load the model which is already trained to make the prediction.

The data is checked to see if there is any error in the format or/and type, then preprocessed and it columns reformated in order to get a matrix with the required size and pased trough our model to get the prediction.

4. app.py
Is the main scripts which controls the web structure using streamlit.

## Author
Jesús Bueno - Project Manager/dev & doc  
