# Parkinson Screening using IMUs data

## Description

This project aims to build multiple classifiers for the
identification of Parkinson Disease symptoms using Machine Learning and provide those results to medical staff.

In this project we are using IMU (Inertial Measurement Unit)
data to facilitate the Parkinson Disease diagnosis.

Here you could find the paper with all the research and design process: [Link]().

Project Implemented as Bachelor's thesis of:

- Jose García
- Julian Bolaños

Mentors:
- Andrés Navarro
- Nicolás Salazar


## Project Structure

This project followed a custom version of CRISP-DM methodology,
and the folder structure is explained below:

`notebooks`: Jupyter notebooks used for experimentation, analysis,
modeling and development.

`data`: Store all the raw, processed and intermediate data from IMUs

`docs`: Documentation from src functions and project in general

`results`: Stores graphics, metrics, weights and model's hyperparameters.

`src`: Python scripts for data download, formating, preprocessing, model building and metrics calculation

`enviroment`: Configuration files, API keys and environment variables.

## Project Configuration

To run any resource from this project we strongly recommend create an enviroment using
our `.yml` file located in `enviroment/` and install all the dependencies on it.