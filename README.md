# Handwritten Digit and Author Recognition

Submitted by: Emily Wang

# Overview:

This project aims to develop a system that recognizes handwritten digits and identifies the author of the handwriting. It involves training two separate models for each task and using RNN and LSTM to enhance handwriting recognition.

# Features

- Task 1: Network design for recognizing the digit from an image.
- Task 2: Network design for identifying the author of the digit.
- Independent Loss Functions: Separate training functions (train()) and loss calculations for both tasks.
- Accuracy Testing: Experiments with different accuracy ratios (0.7, 0.3) to determine optimal performance.
- Excel Result Comparison: Results for both tasks are compiled and analyzed in an Excel spreadsheet.- 
- Image Preprocessing: Preprocessing of images similar to digit classification but adapted for face image data provided by the professor.

# Code Structure

- Main File: Runs the models and generates results. 
- Train Function: Handles the independent training for each task, incorporating separate loss functions.
- RNN/LSTM Models: Utilized for recognizing handwriting pixel by pixel.

# How to Run

- Clone the repository.
- Run the main file to start the training and testing process.
- Check the results in the generated Excel file for accuracy comparison.
