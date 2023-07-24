# Machine Learning Model Comparison

This project compares different machine learning models on the Titanic survival dataset obtained from Kaggle. The models compared in this project include Decision Trees, Support Vector Machine Classifier, Logistic Regression, and K-Nearest Neighbors (KNN). Both an off-the-shelf KNN model and a custom implementation of KNN have been used for this comparison.

## Project Structure
* **dataset**: This directory contains the training and testing datasets. The 'train.csv' file is used for training the classifiers, while the 'test.csv' and 'gender_submission.csv' files are used for evaluating their performance.
* **src**: This directory contains the Python scripts. The 'classifiers.py' file implements the classifiers and evaluates their performance.
* **results**: This directory will contain the output from the classifiers (currently empty).
* **README.md**: This is the file you're currently reading.
* **.gitignore**: This file tells git which files (or patterns) it should ignore.

## Usage
1. Clone the repository: `git clone https://github.com/Zephir0g/ML_project.git`
2. Navigate into the project directory: `cd ML_project/src`
3. Run the Python script to train and evaluate the classifiers: `python classifiers.py`

## Classifiers
The following classifiers are included:
* **DecisionTreeClassifier**: This is an implementation of a decision tree classifier.
* **SVC**: This is an implementation of a Support Vector Machine classifier.
* **LogisticRegression**: This is an implementation of a Logistic Regression classifier.
* **KNeighborsClassifier**: This is an implementation of a K-Nearest Neighbors classifier using scikit-learn.
* **KNN**: This is a custom implementation of a K-Nearest Neighbors classifier.

Each of these classifiers is trained on the provided training data and evaluated on the test data. The performance of each classifier is output to the console, and a ROC curve of each model's performance is displayed.

## Results
After running the classifiers, you will see their accuracy scores printed to the console, as well as a plot displaying the ROC curve for each model. This will give you an understanding of each model's performance on the test data.

Please note that the results may vary each time the script is run, due to the inherent randomness in some of the classifier's algorithms.

This project is perfect for beginners in machine learning, as it provides a clear comparison between different types of classifiers on a well-known dataset. Enjoy exploring and expanding it!

## Authors
[Zephir0g (Bohdan Sukhovarov)](https://github.com/Zephir0g)

[Cyandide Toxicity (Mikhail Semenov)](#)
