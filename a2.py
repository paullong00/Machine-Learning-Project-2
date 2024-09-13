# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:07:13 2023

@author: Paul Long
"""
"""
===========Imports============
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
import time
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC



"""
Task 1
"""

#Read file into Dataframe
file = "fashion-mnist_train.csv"
df = pd.read_csv(file)

#Get sandals, sneakers, and ankle boots by label
required_labels = [5, 7, 9]  
#Create dataframe of all rows with label in selected_labels array
selected_data = df[df['label'].isin(required_labels)]

#Separate labels and feature vectors
labels = selected_data['label']
#Drop label column
features = selected_data.drop('label', axis=1)

#Display one image for each class
fig, axs = plt.subplots(1, len(required_labels), figsize=(10, 4))

for i, label in enumerate(required_labels):
    #Find the first instance of each class
    index = labels[labels == label].index[0]
    image_data = np.array(features.iloc[index]).reshape(28, 28)  # Reshape to 28x28

    #Display Image
    axs[i].imshow(image_data, cmap='gray')
    axs[i].set_title(f"Class {label}")

plt.show()


"""
Task 2
"""

#ml model, feature vectors, labels, num folds, num samples from dataset
def k_fold_cross_validation(classifier, features, labels, k=5, num_samples=None):

    #Shuffle data for repoduction
    kf = KFold(n_splits=k, shuffle=True, random_state=23)

    #Storing data for each fold
    train = []
    prediction = []
    accuracy = []
    confusion_matrices = []

    #Limit number of samples iterated through with num_samples
    #Indexes are assigned to the indices for the current fold
    for train_index, test_index in kf.split(features[:num_samples]):
        #Extract training and testing data
        X_train = features.iloc[train_index]
        X_test = features.iloc[test_index]
        y_train = labels.iloc[train_index]
        y_test  = labels.iloc[test_index]

        #Train the classifier
        start_train_time = time.time()
        classifier.fit(X_train, y_train)
        end_train_time = time.time()
        train_time = end_train_time - start_train_time
        train.append(train_time)

        #Predict on the test set
        start_pred_time = time.time()
        y_pred = classifier.predict(X_test)
        end_pred_time = time.time()
        pred_time = end_pred_time - start_pred_time
        prediction.append(pred_time)

        #Calculate confusion matrix and accuracy
        confusion_mat = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        confusion_matrices.append(confusion_mat)
        accuracy.append(accuracy)

    #Calculate averages, minimums, and maximums
    train_time_per_sample = np.mean(train) / num_samples
    mean_pred_time_per_sample = np.mean(prediction) / len(features)
    mean_accuracy = np.mean(accuracy)

    min_train_time_per_sample = min(train) / num_samples
    max_train_time_per_sample = max(train) / num_samples

    min_pred_time_per_sample = min(prediction) / len(features)
    max_pred_time_per_sample = max(prediction) / len(features)

    #Return tuple containing all calculated data
    return (
        train_time_per_sample,
        mean_pred_time_per_sample,
        mean_accuracy,
        min_train_time_per_sample,
        max_train_time_per_sample,
        min_pred_time_per_sample,
        max_pred_time_per_sample,
        confusion_matrices
    )



"""
Task 3
"""

#Initialize the Perceptron classifier
p = Perceptron()

#Vary the number of samples for evaluation
sample_sizes = [100, 500, 1000, 2000, 5000]

#Lists to store results for plotting
accuracies = []
train = []
prediction = []

for sample_size in sample_sizes:
    #Call validation function from task 2
    result = k_fold_cross_validation(
        p, features, labels, k=5, num_samples=sample_size
    )

    #Access the results: Only train time per sample, prediction time per sample, and accuracy are required for plotting
    (
        train_time_per_sample,
        mean_pred_time_per_sample,
        mean_accuracy,
        _,_,_,_,_,
    ) = result

    accuracies.append(mean_accuracy)
    train.append(train_time_per_sample)
    prediction.append(mean_pred_time_per_sample)

#Plot Time vs Num Samples and Accuracy vs Num Samples
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(sample_sizes, train, marker='o', label='Training Time')
plt.plot(sample_sizes, prediction, marker='o', label='Prediction Time')
plt.xlabel('Number of Samples')
plt.ylabel('Time (seconds)')
plt.title('Training and Prediction Times vs Number of Samples')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(sample_sizes, accuracies, marker='o', label='Accuracy')
plt.xlabel('Number of Samples')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Samples')

plt.tight_layout()
plt.show()

# Print the mean prediction accuracy
print(f"Perceptron Mean Prediction Accuracy: {np.mean(accuracies)}")


"""
Task 4
"""
#Repeat steps of Task 3 using Decision Tree Classifier instead of Perceptron

dt_classifier = DecisionTreeClassifier()

sample_sizes = [100, 500, 1000, 2000, 5000]

accuracies = []
train = []
prediction = []

for sample_size in sample_sizes:
    
    result = k_fold_cross_validation(
        dt_classifier, features, labels, k=5, num_samples=sample_size
    )

    (
        train_time_per_sample,
        mean_pred_time_per_sample,
        mean_accuracy,
        _,_,_,_,_,
    ) = result

    accuracies.append(mean_accuracy)
    train.append(train_time_per_sample)
    prediction.append(mean_pred_time_per_sample)

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(sample_sizes, train, marker='o', label='Training Time')
plt.plot(sample_sizes, prediction, marker='o', label='Prediction Time')
plt.xlabel('Number of Samples')
plt.ylabel('Time (seconds)')
plt.title('Decision Tree: Training and Prediction Times vs Number of Samples')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(sample_sizes, accuracies, marker='o', label='Accuracy')
plt.xlabel('Number of Samples')
plt.ylabel('Accuracy')
plt.title('Decision Tree: Accuracy vs Number of Samples')

plt.tight_layout()
plt.show()


print(f"Decision Tree Mean Prediction Accuracy: {np.mean(accuracies)}")


"""
Task 5
"""
#Similar steps to previous two tasks, taking into account different values for the parameter 'k'

sample_sizes = [100, 500, 1000, 2000, 5000]

#Different choices for the parameter 'k'
k_values = [1, 3, 5, 7, 9]

optimal_k_accuracies = []
optimal_k_train = []
optimal_k_prediction = []

for k in k_values:
    #Initialize classifier with current k value
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    #Lists to store results for the current k
    accuracies = []
    train = []
    prediction = []

    for sample_size in sample_sizes:
        result = k_fold_cross_validation(
            knn_classifier, features, labels, k=5, num_samples=sample_size
        )

        (
            train_time_per_sample,
            mean_pred_time_per_sample,
            mean_accuracy,
            _,_,_,_,_,
        ) = result

        accuracies.append(mean_accuracy)
        train.append(train_time_per_sample)
        prediction.append(mean_pred_time_per_sample)

    #Find the sample size that maximizes accuracy for the current k
    optimal_sample_size_index = np.argmax(accuracies)

    #Store results for the optimal sample size and k
    optimal_k_accuracies.append(accuracies[optimal_sample_size_index])
    optimal_k_train.append(train[optimal_sample_size_index])
    optimal_k_prediction.append(prediction[optimal_sample_size_index])


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(k_values, optimal_k_accuracies, marker='o')
plt.xlabel('k (Number of Neighbours)')
plt.ylabel('Mean Prediction Accuracy')
plt.title('KNN: Mean Prediction Accuracy vs k')

plt.subplot(1, 2, 2)
plt.plot(sample_sizes, optimal_k_train, marker='o', label='Training Time')
plt.plot(sample_sizes, optimal_k_prediction, marker='o', label='Prediction Time')
plt.xlabel('Number of Samples')
plt.ylabel('Time (seconds)')
plt.title('KNN: Training and Prediction Times vs Number of Samples for Optimal k')
plt.legend()

plt.tight_layout()
plt.show()

#Print the best achievable mean prediction accuracy and the corresponding optimal k
best_accuracy = np.max(optimal_k_accuracies)
optimal_k_index = np.argmax(optimal_k_accuracies)
optimal_k = k_values[optimal_k_index]

print(f"KNN Best Mean Prediction Accuracy: {best_accuracy}")
print(f"Optimal k: {optimal_k}")



"""
Task 6
"""

sample_sizes = [100, 500, 1000, 2000, 5000]

#Different choices for the parameter gamma
gamma_values = [1, 3, 5, 7, 9]

#Lists to store results for plotting
optimal_gamma_accuracies = []
optimal_gamma_train = []
optimal_gamma_prediction = []

for gamma in gamma_values:
    #Initialize the SVM classifier with the RBF kernel and the current gamma value
    svm_classifier = SVC(kernel='rbf', gamma=gamma)

    accuracies = []
    train = []
    prediction = []

    for sample_size in sample_sizes:
        result = k_fold_cross_validation(
            svm_classifier, features, labels, k=5, num_samples=sample_size
        )

        (
            train_time_per_sample,
            mean_pred_time_per_sample,
            mean_accuracy,
            _,_,_,_,_,
        ) = result

        accuracies.append(mean_accuracy)
        train.append(train_time_per_sample)
        prediction.append(mean_pred_time_per_sample)

    #Find the sample size that maximizes accuracy for the current gamma
    optimal_sample_size_index = np.argmax(accuracies)

    #Store results for the optimal sample size and gamma
    optimal_gamma_accuracies.append(accuracies[optimal_sample_size_index])
    optimal_gamma_train.append(train[optimal_sample_size_index])
    optimal_gamma_prediction.append(prediction[optimal_sample_size_index])

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(gamma_values, optimal_gamma_accuracies, marker='o')
plt.xlabel('Gamma')
plt.ylabel('Mean Prediction Accuracy')
plt.title('SVM: Mean Prediction Accuracy vs gamma')

plt.subplot(1, 2, 2)
plt.plot(sample_sizes, optimal_gamma_train, marker='o', label='Training Time')
plt.plot(sample_sizes, optimal_gamma_prediction, marker='o', label='Prediction Time')
plt.xlabel('Number of Samples')
plt.ylabel('Time (seconds)')
plt.title('SVM: Training and Prediction Times vs Number of Samples for Optimal Gamma')
plt.legend()

plt.tight_layout()
plt.show()

best_accuracy = np.max(optimal_gamma_accuracies)
optimal_gamma_index = np.argmax(optimal_gamma_accuracies)
optimal_gamma = gamma_values[optimal_gamma_index]

print(f"Task 6 Best Mean Prediction Accuracy: {best_accuracy}")
print(f"Optimal Gamma: {optimal_gamma}")



"""
======================Task 7 answered in PDF Doc=======================
"""