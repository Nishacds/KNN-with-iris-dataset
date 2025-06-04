# KNN-with-iris-dataset

#  K-Nearest Neighbors (KNN) Classification on the Iris Dataset

This project demonstrates a complete end-to-end implementation of the **K-Nearest Neighbors (KNN)** algorithm on the classic **Iris flower classification dataset** using Python and scikit-learn. The workflow includes **data exploration, preprocessing, model training, evaluation**, and **visualization of decision boundaries**.

##  Dataset Overview

The [Iris dataset](https://archive.ics.uci.edu/ml/datasets/Iris) is a widely used multivariate dataset introduced by Sir Ronald A. Fisher. It contains 150 instances of iris flowers, each described by four features:

* Sepal Length (cm)
* Sepal Width (cm)
* Petal Length (cm)
* Petal Width (cm)

There are three target classes:

* **Iris Setosa**
* **Iris Versicolor**
* **Iris Virginica**

##  Tasks Performed

### 1. **Data Loading and Exploration**

* Loaded the dataset using sklearn.datasets.load_iris.
* Displayed feature names and target class labels.
* Converted to a Pandas DataFrame for easy exploration and manipulation.

### 2. **Feature Normalization**

* Standardized the features using StandardScaler to ensure fair distance measurements during KNN classification.

### 3. **Model Training and Evaluation**

* Performed a **train-test split** (80-20) on the normalized data.
* Trained KNeighborsClassifier for different values of **K = \[1, 3, 5, 7, 9]**.
* Evaluated each model using:

  * **Accuracy Score**
  * **Confusion Matrix**
  * **Classification Report** (Precision, Recall, F1-score)

### 4. **Performance Comparison**

* Plotted **accuracy vs K** to identify the optimal number of neighbors for best performance.

### 5. **Decision Boundary Visualization**

* For visual intuition, used only **2 features (petal length and width)** to reduce the feature space to 2D.
* Trained KNN and plotted the **decision boundaries** with a mesh grid.
* Visualized class separation and KNN decision regions using matplotlib.

##  Results

* Achieved **high classification accuracy** across multiple K values.
* **Best performance** observed around K=3 and K=5.
* Decision boundaries clearly illustrate how KNN separates the classes based on proximity in feature space.


##  Key Highlights

*  Hands-on application of **supervised classification**.
*  Demonstrated the impact of **K value selection**.
*  Visualized classifier behavior using 2D projection of the feature space.
*  Reinforced best practices in **data preprocessing** and **model evaluation**.

## Tech Stack

* Language: Python
* **Libraries**: numpy, pandas, matplotlib, seaborn, scikit-learn
* **Model**: K-Nearest Neighbors Classifier (sklearn.neighbors.KNeighborsClassifier)

