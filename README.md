# Machine Learning Model Comparison

This project is about comparing different machine learning models on a dataset obtained from Kaggle. The models we compare are Decision Trees, Support Vector Classifier, Logistic Regression, and K-Nearest Neighbors (KNN).


## Project Structure

The project is organized into Python script:

`main.py`: This script loads the data, preprocesses it, initializes and trains the four classifiers, makes predictions with them and finally, evaluates their performance.

## Dataset

The dataset we use is the Titanic dataset from Kaggle. The dataset is split into a training set and a testing set. The training set contains passenger details and whether they survived or not. The testing set contains passenger details without the survival information, which we have to predict.

## Running the Scripts

Before running the scripts, make sure that you have the necessary Python packages installed. You can install them with pip:

pip install pandas scikit-learn matplotlib

`python main.py`


## Results

The scripts print out the accuracy of each model. In addition, `testimplementation.py` also displays the ROC curves for SVM and Logistic Regression.

Remember to look at the data and the problem you are trying to solve before deciding which model to use. Different models have different strengths and weaknesses and work best for different types of data and different types of problems.
