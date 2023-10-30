# ML_Project1
Machine Learning - Project 1 created by GitHub Classroom

# General Information
This repository contains the code for CS-433 ML_Project1 2023 

## Team members 
The project has been done by the following members :

Shrinidhi Singaravelan \
Salya Amanda Diallo  \
Fanny Sophie Ghez

# Overall Structure of project

## Description 
We can retrieve the data from the GitHub of the course : https://github.com/epfml/ML_course/tree/master/lectures

## Python files used :

*helpers.py* : This is the function given by default, allowing us to download a file and create a submission. 

*implementations.py* : We implement all these various methods given in the project description and more that we ourselves added, such as cross validation methods for ridge regression. 

*clean_and_predict* : This file is used to clean the data by splitting our data into 2 according to the sex of the person, standardize our matrix and add an Intercept. In this file there is also a predict function that returns -1 for all values below a given threshold (1/4) and 1 for all values above the threshold. 

*cross_validation* : This file is used to do cross-validation (CV) on the ridge regression algorithm we implemented. We wanted to compare the result we get with the ones we obtained from setting ourselves the parameters.

*run.ipynb* : This function gives us prediction outputs for both train and test data for each one of the function (LS, MSE,...)

## Predictions :

The folder _predictions_ contains the csv files we obtained when we last ran our notebook.

## Report :
report_ML.pdf : This contains a pdf of a 2-pages report of our 2023 ML project.


