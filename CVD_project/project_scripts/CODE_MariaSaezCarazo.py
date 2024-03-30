# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.models import load_model


#DECISION_TREE
def train_dt(X_train, y_train):
    #Initialize the Decision Tree classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    #Fit the model to the training data
    dt_classifier.fit(X_train, y_train)
    return dt_classifier

def save_dt(dt_classifier, dt_filename):
    #Save trained model
    joblib.dump(dt_classifier, dt_filename)

def load_dt(dt_filename):
    #Load trained model
    return joblib.load(dt_filename)

def variables_dt(dt_classifier, feature_importance):
    #Get feature importances
    importances = dt_classifier.feature_importances_
    #Create a data frame with feature names and their importance scores
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    #Sort the data frame by importance in descending order
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    #Plot the feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='lightgreen')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance for CVD Diagnosis with DT')
    plt.show()

def accuracy_dt(dt_classifier, X_test, y_test):
    #Calculate accuracy on the test set
    y_pred = dt_classifier.predict(X_test)
    return accuracy_score(y_test, y_pred)

def predict_risk_dt(dt_classifier, new_patient):
    #Make predictions on the new patient
    return dt_classifier.predict(new_patient)

def predict_proba_dt(dt_classifier, new_patient):
    #Make predictions on the new patient
    return dt_classifier.predict_proba(new_patient)

def validation_dt(dt_classifier, X_validation, y_validation):
    #Make predictions on the validation set
    probabilities = dt_classifier.predict_proba(X_validation)[:, 1]
    #Convert predictions to binary
    binary_predictions = (probabilities >= 0.5).astype(int)
    #Evaluate predictions
    accuracy_binary = accuracy_score(y_validation, binary_predictions)
    return accuracy_binary


def DTmodel(exists):
    if exists:
        dt_model = load_dt('dt_model.joblib')

    else:
        # Load CSV file
        data = pd.read_csv(r"C:\Users\mario\OneDrive\Escritorio\Uni\SE\SKOVDE\A3\ThesisMethods\cardio_train.csv")
        # Data preprocessing
        data = data[(data['height'] >= 100) & (data['height'] <= 210) & (data['weight'] >= 20) & (data['weight'] <= 250)]
        data = data.dropna()
        #data = data.head(50000)
        data = data.drop('id', axis=1)
        X = data.drop('cardio', axis=1)
        y = data['cardio']

        #Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #Train DT model
        dt_model = train_dt(X_train, y_train)

        #Save DT model
        save_dt(dt_model, 'dt_model.joblib')
    return dt_model

def main_dt():
    
    dt_model = DTmodel(True)

    #Dataframe with 1 patient data
    new_patient = diagnose_patient()
    new_patientB = diagnose_patientB()
    new_patientC = diagnose_patientC()
    new_patientD = diagnose_patientD()

    #Analyze if the patient has risk of CVD with DT model
    risk_disease_dt = predict_risk_dt(dt_model, new_patient)
    print(f"DT, Diagnosis for CVD A: {risk_disease_dt}")
    risk_disease_dtB = predict_risk_dt(dt_model, new_patientB)
    print(f"DT, Diagnosis for CVD B: {risk_disease_dtB}")
    risk_disease_dtC = predict_risk_dt(dt_model, new_patientC)
    print(f"DT, Diagnosis for CVD C: {risk_disease_dtC}")
    risk_disease_dtD = predict_risk_dt(dt_model, new_patientD)
    print(f"DT, Diagnosis for CVD D: {risk_disease_dtD}")

    #Analyze probability of disease with DT model
    probabilities_dt = predict_proba_dt(dt_model, new_patient)
    print(f"DT, Probability of having cardiovascular disease A: {probabilities_dt}")
    probabilities_dtB = predict_proba_dt(dt_model, new_patientB)
    print(f"DT, Probability of having cardiovascular disease B: {probabilities_dtB}")
    probabilities_dtC = predict_proba_dt(dt_model, new_patientC)
    print(f"DT, Probability of having cardiovascular disease C: {probabilities_dtC}")
    probabilities_dtD = predict_proba_dt(dt_model, new_patientD)
    print(f"DT, Probability of having cardiovascular disease D: {probabilities_dtD}")

    #Analyze feature importance for DT model
    features = X.columns
    variables_dt(dt_model, features)

    #Analyze accuracy of DT model
    dt_model_accuracy = accuracy_dt(dt_model, X_test, y_test)
    print(f"DT Accuracy on the test set: {dt_model_accuracy}")

    #Check accuracy of the validation set diagnosis
    valid_dt = validation_dt(dt_model, X_validation, y_validation)
    valid_dt2 = validation_dt(dt_model, X_validation2, y_validation2)
    valid_dt3 = validation_dt(dt_model, X_validation3, y_validation3)

    print(f"DT, Accuracy of the validation set 1 diagnosis: {valid_dt}")
    print(f"DT, Accuracy of the validation set 2 diagnosis: {valid_dt2}")
    print(f"DT, Accuracy of the validation set 3 diagnosis: {valid_dt3}")


#RANDOM_FOREST
def train_rf(X_train, y_train):
    #Initialize the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    #Fit the model to the training data
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

def save_rf(rf_classifier, rf_filename):
    #Save trained model
    joblib.dump(rf_classifier, rf_filename)

def load_rf(rf_filename):
    #Load saved model
    return joblib.load(rf_filename)

def variables_rf(rf_classifier, feature_importance):
    #Get feature importances
    importances = rf_classifier.feature_importances_
    #Create a data frame with feature names and their importance scores
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    #Sort the data frame by importance in descending order
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    #Plot the feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='lightgreen')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance for CVD Diagnosis with RF')
    plt.show()

def accuracy_rf(rf_classifier, X_test, y_test):
    #Calculate accuracy on the test set
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def predict_risk_rf(rf_classifier, new_patient):
    #Make predictions on the new patient
    return rf_classifier.predict(new_patient)

def predict_proba_rf(rf_classifier, new_patient):
    #Make predictions on the new patient
    return rf_classifier.predict_proba(new_patient)

def validation_rf(rf_classifier, X_validation, y_validation):
    #Make predictions on the validatiom set
    probabilities = rf_classifier.predict_proba(X_validation)[:, 1]
    #Convert predictions to binary
    binary_predictions = (probabilities >= 0.5).astype(int)
    #Evaluate predictions
    accuracy_binary = accuracy_score(y_validation, binary_predictions)
    return accuracy_binary

def RFmodel(exists):
    if exists:
        rf_model = load_rf('rf_model.joblib')

    else:
        #Load CSV file
        data = pd.read_csv(r"C:\Users\mario\OneDrive\Escritorio\Uni\SE\SKOVDE\A3\ThesisMethods\cardio_train.csv")
        
        #Data preprocessing
        data = data[(data['height'] >= 100) & (data['height'] <= 210) & (data['weight'] >= 20) & (data['weight'] <= 250)]
        data = data.dropna()
        #data = data.head(50000)
        data = data.drop('id', axis=1)
        X = data.drop('cardio', axis=1)
        y = data['cardio']

        #Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #Train RF model
        rf_model = train_rf(X_train, y_train)

        #Save RF model
        save_rf(rf_model, 'rf_model.joblib')
    return rf_model

def main_rf():
    
    rf_model = RFmodel(True)

    #Dataframe with 1 patient data
    new_patient = diagnose_patient()
    new_patientB = diagnose_patientB()
    new_patientC = diagnose_patientC()
    new_patientD = diagnose_patientD()

    #Analyze if the patient has risk of CVD with RF model
    risk_disease_rf = predict_risk_rf(rf_model, new_patient)
    print(f"RF, Diagnosis for CVD A: {risk_disease_rf}")
    risk_disease_rfB = predict_risk_dt(rf_model, new_patientB)
    print(f"RF, Diagnosis for CVD B: {risk_disease_rfB}")
    risk_disease_rfC = predict_risk_dt(rf_model, new_patientC)
    print(f"RF, Diagnosis for CVD C: {risk_disease_rfC}")
    risk_disease_rfD = predict_risk_dt(rf_model, new_patientD)
    print(f"RF, Diagnosis for CVD D: {risk_disease_rfD}")

    #Analyze probability of disease with RF model
    probabilities_rf = predict_proba_rf(rf_model, new_patient)
    print(f"RF, Probability of having cardiovascular disease A: {probabilities_rf}")
    probabilities_rfB = predict_proba_rf(rf_model, new_patientB)
    print(f"RF, Probability of having cardiovascular disease B: {probabilities_rfB}")
    probabilities_rfC = predict_proba_rf(rf_model, new_patientC)
    print(f"RF, Probability of having cardiovascular disease C: {probabilities_rfC}")
    probabilities_rfD = predict_proba_rf(rf_model, new_patientD)
    print(f"RF, Probability of having cardiovascular disease D: {probabilities_rfD}")

    #Analyze feature importance for RF model
    features = X.columns
    variables_rf(rf_model, features)

    #Analyze accuracy of RF model
    rf_model_accuracy = accuracy_rf(rf_model, X_test, y_test)
    print(f"RF Accuracy on the test set: {rf_model_accuracy}")

    #Check accuracy of the validation set diagnosis
    valid_rf = validation_rf(rf_model, X_validation, y_validation)
    valid_rf2 = validation_rf(rf_model, X_validation2, y_validation2)
    valid_rf3 = validation_rf(rf_model, X_validation3, y_validation3)

    print(f"RF, Accuracy of the validation set 1 diagnosis: {valid_rf}")
    print(f"RF, Accuracy of the validation set 2 diagnosis: {valid_rf2}")
    print(f"RF, Accuracy of the validation set 3 diagnosis: {valid_rf3}")

#CNN
def train_cnn(X_train_scaled, y_train, X_test_scaled, y_test):
    cnn_classifier = Sequential()
    cnn_classifier.add(Conv1D(filters=X_train_scaled.shape[1], kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)))
    cnn_classifier.add(MaxPooling1D(pool_size=2))
    cnn_classifier.add(Flatten())
    cnn_classifier.add(Dense(64, activation='relu'))
    cnn_classifier.add(Dense(1, activation='sigmoid'))
    #Compile model
    cnn_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #Train model
    cnn_classifier.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))
    return cnn_classifier

def save_cnn(cnn_classifier, filename):
    #Save trained model
    saved = cnn_classifier.save(filename)
    return saved

def load_cnn(filename):
    #Load saved model
    return load_model(filename)

def variables_cnn(cnn_classifier, X_test_reshaped, X_train_scaled):
    #Mean absolute value of the filters in the first layer
    layer_output = Model(inputs=cnn_classifier.input, outputs=cnn_classifier.layers[0].output)
    filters = layer_output.predict(X_test_reshaped)
    #Calculate mean absolute value for each filter
    mean_abs_values = np.mean(np.abs(filters), axis=(0, 1))
    sorted_indices = np.argsort(mean_abs_values)[::-1]
    sorted_mean_abs_values = mean_abs_values[sorted_indices]
    #Plot mean absolute values
    plt.barh(range(len(sorted_mean_abs_values)), sorted_mean_abs_values, color='lightgreen')
    plt.yticks(range(X_train_scaled.shape[1]), [X.columns[i] for i in sorted_indices])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance for CVD Diagnosis with CNN')
    plt.show()

def accuracy_cnn(cnn_classifier, X_test_reshaped, y_test):
    #Calculate accuracy on the test set
    loss, accuracy = cnn_classifier.evaluate(X_test_reshaped, y_test)
    return accuracy

def risk_cnn(cnn_classifier, new_patient_data_reshaped):
    #Make predictions on the new patient
    prediction_prob = cnn_classifier.predict(new_patient_data_reshaped)
    #Interpret the prediction
    predicted_class = 1 if prediction_prob[0][0] >= 0.5 else 0
    return predicted_class

def prob_cnn(cnn_classifier,new_patient_data_reshaped):
    #Make predictions on the new patient
    prediction_prob = cnn_classifier.predict(new_patient_data_reshaped)
    return prediction_prob[0][0]

def validation_cnn(cnn_classifier, X_validation_reshaped, y_validation):
    #Make predictions on the validation set
    y_pred = cnn_classifier.predict(X_validation_reshaped)
    #Convert predictions to binary
    y_pred_binary = (y_pred > 0.5).astype(int)
    #Flatten true labels for comparison
    y_test_flat = y_validation.values.flatten()
    #Evaluate predictions
    accuracy = accuracy_score(y_test_flat, y_pred_binary)
    return accuracy

def CNNmodel(exists):
    if exists:
        cnn_model = load_cnn('cnn_model.h5')
    else:
        #Load CSV file
        data = pd.read_csv(r"C:\Users\mario\OneDrive\Escritorio\Uni\SE\SKOVDE\A3\ThesisMethods\cardio_train.csv")
        
        #Preprocess data
        data = data[(data['height'] >= 100) & (data['height'] <= 210) & (data['weight'] >= 20) & (data['weight'] <= 250)]
        data = data.dropna()
        #data = data.head(50000)
        data = data.drop('id', axis=1)
        X = data.drop('cardio', axis=1)
        y = data['cardio']

        #Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #Standardize input features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        #Train CNN model
        cnn_model = train_cnn(X_train_scaled, y_train, X_test_scaled, y_test)

        #Save CNN model
        save_cnn(cnn_model, 'cnn_model.h5')
    return cnn_model

def main_cnn():

    cnn_model = CNNmodel(True)

    #New patient data in numpy array
    new_patient_data = np.array([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
    new_patient_dataB = np.array([[ageB, genderB, heightB, weightB, ap_hiB, ap_loB, cholesterolB, glucB, smokeB, alcoB, activeB]])
    new_patient_dataC = np.array([[ageC, genderC, heightC, weightC, ap_hiC, ap_loC, cholesterolC, glucC, smokeC, alcoC, activeC]])
    new_patient_dataD = np.array([[ageD, genderD, heightD, weightD, ap_hiD, ap_loD, cholesterolD, glucD, smokeD, alcoD, activeD]])

    #Standardize new patient data
    new_patient_data_scaled = scaler.transform(new_patient_data)
    new_patient_data_scaledB = scaler.transform(new_patient_dataB)
    new_patient_data_scaledC = scaler.transform(new_patient_dataC)
    new_patient_data_scaledD = scaler.transform(new_patient_dataD)

    #Reshape input data for the CNN
    new_patient_data_reshaped = new_patient_data_scaled.reshape(1, new_patient_data_scaled.shape[1], 1)
    new_patient_data_reshapedB = new_patient_data_scaledB.reshape(1, new_patient_data_scaledB.shape[1], 1)
    new_patient_data_reshapedC = new_patient_data_scaledC.reshape(1, new_patient_data_scaledC.shape[1], 1)
    new_patient_data_reshapedD = new_patient_data_scaledD.reshape(1, new_patient_data_scaledD.shape[1], 1)


    #Analyze if the patient has risk of CVD with CNN model
    risk_disease_cnn = risk_cnn(cnn_model, new_patient_data_reshaped)
    print(f"CNN, Diagnosis for CVD A: {risk_disease_cnn}")
    risk_disease_cnnB = risk_cnn(cnn_model, new_patient_data_reshapedB)
    print(f"CNN, Diagnosis for CVD B: {risk_disease_cnnB}")
    risk_disease_cnnC = risk_cnn(cnn_model, new_patient_data_reshapedC)
    print(f"CNN, Diagnosis for CVD C: {risk_disease_cnnC}")
    risk_disease_cnnD = risk_cnn(cnn_model, new_patient_data_reshapedD)
    print(f"CNN, Diagnosis for CVD D: {risk_disease_cnnD}")

    #Analyze probability of disease with CNN model
    probabilities_cnn = prob_cnn(cnn_model, new_patient_data_reshaped)
    print(f"CNN, Probability of having cardiovascular disease A: {probabilities_cnn}")
    probabilities_cnnB = prob_cnn(cnn_model, new_patient_data_reshapedB)
    print(f"CNN, Probability of having cardiovascular disease B: {probabilities_cnnB}")
    probabilities_cnnC = prob_cnn(cnn_model, new_patient_data_reshapedC)
    print(f"CNN, Probability of having cardiovascular disease C: {probabilities_cnnC}")
    probabilities_cnnD = prob_cnn(cnn_model, new_patient_data_reshapedD)
    print(f"CNN, Probability of having cardiovascular disease D: {probabilities_cnnD}")

    #Analyze feature importance for CNN model
    variables_cnn(cnn_model, X_test_reshaped, X_train_scaled)

    #Analyze accuracy of CNN model
    cnn_model_accuracy = accuracy_cnn(cnn_model, X_test_reshaped, y_test)
    print(f"CNN Accuracy on the test set: {cnn_model_accuracy}")

    #Check accuracy of the validation set diagnosis
    valid_cnn = validation_cnn(cnn_model, X_validation_reshaped, y_validation)
    valid_cnn2 = validation_cnn(cnn_model, X_validation_reshaped2, y_validation2)
    valid_cnn3 = validation_cnn(cnn_model, X_validation_reshaped3, y_validation3)

    print(f"CNN, Accuracy of the validation set 1 diagnosis: {valid_cnn:.4f}")
    print(f"CNN, Accuracy of the validation set 2 diagnosis: {valid_cnn2:.4f}")
    print(f"CNN, Accuracy of the validation set 3 diagnosis: {valid_cnn3:.4f}")


def train_ann(X_train_scaled, y_train):
    #Create and train the MLPClassifier
    ann_classifier = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam', random_state=42, max_iter=1000, tol=1e-4)
    ann_classifier.fit(X_train_scaled, y_train)
    return ann_classifier

def save_ann(ann_classifier, filename):
    #Save trained model
    saved = joblib.dump(ann_classifier, filename)
    return saved

def load_ann(filename):
    #Load saved model
    return joblib.load(filename)

def variables_ann(ann_classifier, X_train_scaled):
    #Visualize the importance of features
    weights_input_hidden = ann_classifier.coefs_[0].T
    feature_importance = np.abs(weights_input_hidden).mean(axis=0)

    #Sort feature importance values and corresponding feature names
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_feature_importance = feature_importance[sorted_indices]
    sorted_feature_names = X.columns[sorted_indices]

    #Plot the feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(X_train_scaled.shape[1]), sorted_feature_importance, color='lightgreen')
    plt.yticks(range(X_train_scaled.shape[1]), sorted_feature_names)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance in Artificial Neural Network (ANN)')
    plt.show()

def accuracy_ann(ann_classifier, X_test_scaled, y_test):
    #Calculate accuracy on the test set
    accuracy = ann_classifier.score(X_test_scaled, y_test)
    return accuracy

def risk_ann(ann_classifier, new_patient_data):
    new_patient_data_scaled = scaler.transform(new_patient_data)
    #Make predictions on the new patient
    prediction = ann_classifier.predict(new_patient_data_scaled)
    return prediction[0]

def prob_ann(ann_classifier, new_patient_data):
    new_patient_data_scaled = scaler.transform(new_patient_data)
    #Make predictions on the new patient
    probability = ann_classifier.predict_proba(new_patient_data_scaled)[:, 1]
    return probability[0]

def validation_ann(ann_classifier, X_validation, y_validation):
    #Make predictions on the validation set
    predictions = ann_classifier.predict(X_validation)
    #Evaluate predictions
    accuracy = accuracy_score(y_validation, predictions)
    return accuracy

def ANNmodel(exists):
    if exists:
        ann_model = load_ann('ann_model.joblib')
    else:
        #Load CSV file
        data = pd.read_csv(r"C:\Users\mario\OneDrive\Escritorio\Uni\SE\SKOVDE\A3\ThesisMethods\cardio_train.csv")
        
        #Preprocess data
        data = data[(data['height'] >= 100) & (data['height'] <= 210) & (data['weight'] >= 20) & (data['weight'] <= 250)]
        data = data.dropna()
        #data = data.head(50000)
        data = data.drop('id', axis=1)
        X = data.drop('cardio', axis=1)
        y = data['cardio']

        #Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #Standardize input features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        #Train ANN model
        ann_model = train_ann(X_train_scaled, y_train)

        #Save ANN model
        save_ann(ann_model, 'ann_model.joblib')
    return ann_model

def main_ann():

    ann_model = ANNmodel(True)

    #New patient data in numpy array
    new_patient_data = np.array([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
    new_patient_dataB = np.array([[ageB, genderB, heightB, weightB, ap_hiB, ap_loB, cholesterolB, glucB, smokeB, alcoB, activeB]])
    new_patient_dataC = np.array([[ageC, genderC, heightC, weightC, ap_hiC, ap_loC, cholesterolC, glucC, smokeC, alcoC, activeC]])
    new_patient_dataD = np.array([[ageD, genderD, heightD, weightD, ap_hiD, ap_loD, cholesterolD, glucD, smokeD, alcoD, activeD]])

    #Analyze if the patient has risk of CVD with ANN model
    risk_disease_ann = risk_ann(ann_model, new_patient_data)
    print(f"ANN, Diagnosis for CVD A: {risk_disease_ann}")
    risk_disease_annB = risk_ann(ann_model, new_patient_dataB)
    print(f"ANN, Diagnosis for CVD B: {risk_disease_annB}")
    risk_disease_annC = risk_ann(ann_model, new_patient_dataC)
    print(f"ANN, Diagnosis for CVD C: {risk_disease_annC}")
    risk_disease_annD = risk_ann(ann_model, new_patient_dataD)
    print(f"ANN, Diagnosis for CVD D: {risk_disease_annD}")

    #Analyze probability of disease with SVM model
    probabilities_ann = prob_ann(ann_model, new_patient_data)
    print(f"ANN, Probability of having cardiovascular disease A: {probabilities_ann}")
    probabilities_annB = prob_ann(ann_model, new_patient_dataB)
    print(f"ANN, Probability of having cardiovascular disease B: {probabilities_annB}")
    probabilities_annC = prob_ann(ann_model, new_patient_dataC)
    print(f"ANN, Probability of having cardiovascular disease C: {probabilities_annC}")
    probabilities_annD = prob_ann(ann_model, new_patient_dataD)
    print(f"ANN, Probability of having cardiovascular disease D: {probabilities_annD}")

    #Analyze feature importance for ANN model
    variables_ann(ann_model, X_train_scaled)

    #Analyze accuracy of ANN model
    ann_model_accuracy = accuracy_ann(ann_model, X_test_scaled, y_test)
    print(f"ANN Accuracy on the test set: {ann_model_accuracy}")

    #Check accuracy of the validation set diagnosis
    valid_ann = validation_ann(ann_model, X_validation, y_validation)
    valid_ann2 = validation_ann(ann_model, X_validation2, y_validation2)
    valid_ann3 = validation_ann(ann_model, X_validation3, y_validation3)

    print(f"ANN, Accuracy of the validation set 1 diagnosis: {valid_ann:.4f}")
    print(f"ANN, Accuracy of the validation set 2 diagnosis: {valid_ann2:.4f}")
    print(f"ANN, Accuracy of the validation set 3 diagnosis: {valid_ann3:.4f}")

#New patients data
age = 48
gender = 2
height = 170
weight = 75
ap_hi = 100
ap_lo = 70
cholesterol = 1
gluc = 1
smoke = 0  
alco = 0  
active = 0

ageB = 60
genderB = 1
heightB = 150
weightB = 88
ap_hiB = 120
ap_loB = 80
cholesterolB = 3
glucB = 1
smokeB = 0  
alcoB = 0  
activeB = 1

ageC = 62
genderC = 1
heightC = 158
weightC = 60
ap_hiC = 140
ap_loC = 90
cholesterolC = 1
glucC = 1
smokeC = 0  
alcoC = 0  
activeC = 1

ageD = 48
genderD = 2
heightD = 169
weightD = 82
ap_hiD = 150
ap_loD = 100
cholesterolD = 1
glucD = 1
smokeD = 0  
alcoD = 0  
activeD = 1

#New dataframe with input values for each variable of a new patient
def diagnose_patient():

    new_patient = pd.DataFrame({
        'age': [48],
        'gender': [2],
        'height': [170],
        'weight': [75],
        'ap_hi': [100],
        'ap_lo': [70],
        'cholesterol': [1],
        'gluc': [1],
        'smoke': [0],
        'alco': [0],
        'active': [0]
    })

    return new_patient

def diagnose_patientB():

    new_patient = pd.DataFrame({
        'age': [60],
        'gender': [1],
        'height': [150],
        'weight': [88],
        'ap_hi': [120],
        'ap_lo': [80],
        'cholesterol': [3],
        'gluc': [1],
        'smoke': [0],
        'alco': [0],
        'active': [1]
    })

    return new_patient

def diagnose_patientC():

    new_patient = pd.DataFrame({
        'age': [62],
        'gender': [1],
        'height': [158],
        'weight': [60],
        'ap_hi': [140],
        'ap_lo': [90],
        'cholesterol': [1],
        'gluc': [1],
        'smoke': [0],
        'alco': [0],
        'active': [1]
    })

    return new_patient

def diagnose_patientD():

    new_patient = pd.DataFrame({
        'age': [48],
        'gender': [2],
        'height': [169],
        'weight': [82],
        'ap_hi': [150],
        'ap_lo': [100],
        'cholesterol': [1],
        'gluc': [1],
        'smoke': [0],
        'alco': [0],
        'active': [1]
    })

    return new_patient

#Load CSV file
data = pd.read_csv(r"C:\Users\mario\OneDrive\Escritorio\Uni\SE\SKOVDE\A3\ThesisMethods\cardio_train.csv")

#Data preprocessing
data = data[(data['height'] >= 100) & (data['height'] <= 210) & (data['weight'] >= 20) & (data['weight'] <= 250)]
data = data.dropna()
data_val = data.sample(2000)
data_val2 = data.sample(2000)
data_val3 = data.sample(2000)
#data = data.head(15000)
data = data.drop('id', axis=1)
data_val = data_val.drop('id', axis=1)
data_val2 = data_val2.drop('id', axis=1)
data_val3 = data_val3.drop('id', axis=1)

X_validation = data_val.drop('cardio', axis=1)
y_validation = data_val['cardio']
X_validation2 = data_val2.drop('cardio', axis=1)
y_validation2 = data_val2['cardio']
X_validation3 = data_val3.drop('cardio', axis=1)
y_validation3 = data_val3['cardio']
X = data.drop('cardio', axis=1)
y = data['cardio']

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Standardize input features
scaler = StandardScaler()
X_validation_scaled = scaler.fit_transform(X_validation)
X_validation_scaled2 = scaler.fit_transform(X_validation2)
X_validation_scaled3 = scaler.fit_transform(X_validation3)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Reshape data for 1D CNN
X_validation_reshaped = X_validation_scaled.reshape(X_validation_scaled.shape[0], X_validation_scaled.shape[1], 1)
X_validation_reshaped2 = X_validation_scaled2.reshape(X_validation_scaled2.shape[0], X_validation_scaled2.shape[1], 1)
X_validation_reshaped3 = X_validation_scaled3.reshape(X_validation_scaled3.shape[0], X_validation_scaled3.shape[1], 1)

X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)


if __name__ == "__main__":
    main_rf()
