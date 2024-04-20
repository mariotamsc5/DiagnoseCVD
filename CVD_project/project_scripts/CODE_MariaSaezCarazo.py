# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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
        dt_model = load_dt('dt_model_70000.joblib')

    else:
        
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
    new_patientE = diagnose_patientE()
    new_patientF = diagnose_patientF()
    new_patientG = diagnose_patientG()
    new_patientH = diagnose_patientH()
    new_patientI = diagnose_patientI()
    new_patientJ = diagnose_patientJ()

    #Analyze if the patient has risk of CVD with DT model
    risk_disease_dt = predict_risk_dt(dt_model, new_patient)
    print(f"DT, Disease risk A: {risk_disease_dt}")
    risk_disease_dtB = predict_risk_dt(dt_model, new_patientB)
    print(f"DT, Disease risk B: {risk_disease_dtB}")
    risk_disease_dtC = predict_risk_dt(dt_model, new_patientC)
    print(f"DT, Disease risk C: {risk_disease_dtC}")
    risk_disease_dtD = predict_risk_dt(dt_model, new_patientD)
    print(f"DT, Disease risk D: {risk_disease_dtD}")
    risk_disease_dtE = predict_risk_dt(dt_model, new_patientE)
    print(f"DT, Disease risk E: {risk_disease_dtE}")
    risk_disease_dtF = predict_risk_dt(dt_model, new_patientF)
    print(f"DT, Disease risk F: {risk_disease_dtF}")
    risk_disease_dtG = predict_risk_dt(dt_model, new_patientG)
    print(f"DT, Disease risk G: {risk_disease_dtG}")
    risk_disease_dtH = predict_risk_dt(dt_model, new_patientH)
    print(f"DT, Disease risk H: {risk_disease_dtH}")
    risk_disease_dtI = predict_risk_dt(dt_model, new_patientI)
    print(f"DT, Disease risk I: {risk_disease_dtI}")
    risk_disease_dtJ = predict_risk_dt(dt_model, new_patientJ)
    print(f"DT, Disease risk J: {risk_disease_dtJ}")

    #Analyze probability of disease with DT model
    probabilities_dt = predict_proba_dt(dt_model, new_patient)
    print(f"DT, Probability of having cardiovascular disease A: {probabilities_dt}")
    probabilities_dtB = predict_proba_dt(dt_model, new_patientB)
    print(f"DT, Probability of having cardiovascular disease B: {probabilities_dtB}")
    probabilities_dtC = predict_proba_dt(dt_model, new_patientC)
    print(f"DT, Probability of having cardiovascular disease C: {probabilities_dtC}")
    probabilities_dtD = predict_proba_dt(dt_model, new_patientD)
    print(f"DT, Probability of having cardiovascular disease D: {probabilities_dtD}")
    probabilities_dtE = predict_proba_dt(dt_model, new_patientE)
    print(f"DT, Probability of having cardiovascular disease E: {probabilities_dtE}")
    probabilities_dtF = predict_proba_dt(dt_model, new_patientF)
    print(f"DT, Probability of having cardiovascular disease F: {probabilities_dtF}")
    probabilities_dtG = predict_proba_dt(dt_model, new_patientG)
    print(f"DT, Probability of having cardiovascular disease G: {probabilities_dtG}")
    probabilities_dtH = predict_proba_dt(dt_model, new_patientH)
    print(f"DT, Probability of having cardiovascular disease H: {probabilities_dtH}")
    probabilities_dtI = predict_proba_dt(dt_model, new_patientI)
    print(f"DT, Probability of having cardiovascular disease I: {probabilities_dtI}")
    probabilities_dtJ = predict_proba_dt(dt_model, new_patientJ)
    print(f"DT, Probability of having cardiovascular disease J: {probabilities_dtJ}")


    #Analyze feature importance for DT model
    features = X.columns
    variables_dt(dt_model, features)

    #Analyze accuracy of DT model
    dt_model_accuracy = accuracy_dt(dt_model, X_test, y_test)
    print(f"DT Accuracy on the test set: {dt_model_accuracy}")

    #Check accuracy of the validation set predictions
    valid_dt = validation_dt(dt_model, X_validation, y_validation)
    valid_dt2 = validation_dt(dt_model, X_validation2, y_validation2)
    valid_dt3 = validation_dt(dt_model, X_validation3, y_validation3)

    print(f"DT, Accuracy of the validation set 1 predictions: {valid_dt}")
    print(f"DT, Accuracy of the validation set 2 predictions: {valid_dt2}")
    print(f"DT, Accuracy of the validation set 3 predictions: {valid_dt3}")


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
    return accuracy_score(y_test, y_pred)

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
        rf_model = load_rf('rf_model_70000.joblib')

    else:
        #Train RF model
        rf_model = train_rf(X_train, y_train)

        #Save RF model
        save_rf(rf_model, 'rf_model.joblib')
    return rf_model

def main_rf():
    
    rf_model = RFmodel(False)

    #Dataframe with 1 patient data
    new_patient = diagnose_patient()
    new_patientB = diagnose_patientB()
    new_patientC = diagnose_patientC()
    new_patientD = diagnose_patientD()
    new_patientE = diagnose_patientE()
    new_patientF = diagnose_patientF()
    new_patientG = diagnose_patientG()
    new_patientH = diagnose_patientH()
    new_patientI = diagnose_patientI()
    new_patientJ = diagnose_patientJ()

    #Analyze if the patient has risk of CVD with RF model
    risk_disease_rf = predict_risk_rf(rf_model, new_patient)
    print(f"RF, Disease risk A: {risk_disease_rf}")
    risk_disease_rfB = predict_risk_rf(rf_model, new_patientB)
    print(f"RF, Disease risk B: {risk_disease_rfB}")
    risk_disease_rfC = predict_risk_rf(rf_model, new_patientC)
    print(f"RF, Disease risk C: {risk_disease_rfC}")
    risk_disease_rfD = predict_risk_rf(rf_model, new_patientD)
    print(f"RF, Disease risk D: {risk_disease_rfD}")
    risk_disease_rfE = predict_risk_rf(rf_model, new_patientE)
    print(f"RF, Disease risk E: {risk_disease_rfE}")
    risk_disease_rfF = predict_risk_rf(rf_model, new_patientF)
    print(f"RF, Disease risk F: {risk_disease_rfF}")
    risk_disease_rfG = predict_risk_rf(rf_model, new_patientG)
    print(f"RF, Disease risk G: {risk_disease_rfG}")
    risk_disease_rfH = predict_risk_rf(rf_model, new_patientH)
    print(f"RF, Disease risk H: {risk_disease_rfH}")
    risk_disease_rfI = predict_risk_rf(rf_model, new_patientI)
    print(f"RF, Disease risk I: {risk_disease_rfI}")
    risk_disease_rfJ = predict_risk_rf(rf_model, new_patientJ)
    print(f"RF, Disease risk J: {risk_disease_rfJ}")

    #Analyze probability of disease with RF model
    probabilities_rf = predict_proba_rf(rf_model, new_patient)
    print(f"RF, Probability of having cardiovascular disease A: {probabilities_rf}")
    probabilities_rfB = predict_proba_rf(rf_model, new_patientB)
    print(f"RF, Probability of having cardiovascular disease B: {probabilities_rfB}")
    probabilities_rfC = predict_proba_rf(rf_model, new_patientC)
    print(f"RF, Probability of having cardiovascular disease C: {probabilities_rfC}")
    probabilities_rfD = predict_proba_rf(rf_model, new_patientD)
    print(f"RF, Probability of having cardiovascular disease D: {probabilities_rfD}")
    probabilities_rfE = predict_proba_rf(rf_model, new_patientE)
    print(f"RF, Probability of having cardiovascular disease E: {probabilities_rfE}")
    probabilities_rfF = predict_proba_rf(rf_model, new_patientF)
    print(f"RF, Probability of having cardiovascular disease F: {probabilities_rfF}")
    probabilities_rfG = predict_proba_rf(rf_model, new_patientG)
    print(f"RF, Probability of having cardiovascular disease G: {probabilities_rfG}")
    probabilities_rfH = predict_proba_rf(rf_model, new_patientH)
    print(f"RF, Probability of having cardiovascular disease H: {probabilities_rfH}")
    probabilities_rfI = predict_proba_rf(rf_model, new_patientI)
    print(f"RF, Probability of having cardiovascular disease I: {probabilities_rfI}")
    probabilities_rfJ = predict_proba_rf(rf_model, new_patientJ)
    print(f"RF, Probability of having cardiovascular disease J: {probabilities_rfJ}")


    #Analyze feature importance for RF model
    features = X.columns
    variables_rf(rf_model, features)

    #Analyze accuracy of RF model
    rf_model_accuracy = accuracy_rf(rf_model, X_test, y_test)
    print(f"RF Accuracy on the test set: {rf_model_accuracy}")

    #Check accuracy of the validation set predictions
    valid_rf = validation_rf(rf_model, X_validation, y_validation)
    valid_rf2 = validation_rf(rf_model, X_validation2, y_validation2)
    valid_rf3 = validation_rf(rf_model, X_validation3, y_validation3)

    print(f"RF, Accuracy of the validation set 1 predictions: {valid_rf}")
    print(f"RF, Accuracy of the validation set 2 predictions: {valid_rf2}")
    print(f"RF, Accuracy of the validation set 3 predictions: {valid_rf3}")

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
    cnn_classifier.fit(X_train_scaled, y_train, epochs=20, batch_size=128, validation_data=(X_test_scaled, y_test))
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
        cnn_model = load_cnn('cnn_model_70000.h5')
    else:

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
    new_patient_dataE = np.array([[ageE, genderE, heightE, weightE, ap_hiE, ap_loE, cholesterolE, glucE, smokeE, alcoE, activeE]])
    new_patient_dataF = np.array([[ageF, genderF, heightF, weightF, ap_hiF, ap_loF, cholesterolF, glucF, smokeF, alcoF, activeF]])
    new_patient_dataG = np.array([[ageG, genderG, heightG, weightG, ap_hiG, ap_loG, cholesterolG, glucG, smokeG, alcoG, activeG]])
    new_patient_dataH = np.array([[ageH, genderH, heightH, weightH, ap_hiH, ap_loH, cholesterolH, glucH, smokeH, alcoH, activeH]])
    new_patient_dataI = np.array([[ageI, genderI, heightI, weightI, ap_hiI, ap_loI, cholesterolI, glucI, smokeI, alcoI, activeI]])
    new_patient_dataJ = np.array([[ageJ, genderJ, heightJ, weightJ, ap_hiJ, ap_loJ, cholesterolJ, glucJ, smokeJ, alcoJ, activeJ]])

    #Standardize new patient data
    new_patient_data_scaled = scaler.transform(new_patient_data)
    new_patient_data_scaledB = scaler.transform(new_patient_dataB)
    new_patient_data_scaledC = scaler.transform(new_patient_dataC)
    new_patient_data_scaledD = scaler.transform(new_patient_dataD)
    new_patient_data_scaledE = scaler.transform(new_patient_dataE)
    new_patient_data_scaledF = scaler.transform(new_patient_dataF)
    new_patient_data_scaledG = scaler.transform(new_patient_dataG)
    new_patient_data_scaledH = scaler.transform(new_patient_dataH)
    new_patient_data_scaledI = scaler.transform(new_patient_dataI)
    new_patient_data_scaledJ = scaler.transform(new_patient_dataJ)

    #Reshape input data for the CNN
    new_patient_data_reshaped = new_patient_data_scaled.reshape(1, new_patient_data_scaled.shape[1], 1)
    new_patient_data_reshapedB = new_patient_data_scaledB.reshape(1, new_patient_data_scaledB.shape[1], 1)
    new_patient_data_reshapedC = new_patient_data_scaledC.reshape(1, new_patient_data_scaledC.shape[1], 1)
    new_patient_data_reshapedD = new_patient_data_scaledD.reshape(1, new_patient_data_scaledD.shape[1], 1)
    new_patient_data_reshapedE = new_patient_data_scaledE.reshape(1, new_patient_data_scaledE.shape[1], 1)
    new_patient_data_reshapedF = new_patient_data_scaledF.reshape(1, new_patient_data_scaledF.shape[1], 1)
    new_patient_data_reshapedG = new_patient_data_scaledG.reshape(1, new_patient_data_scaledG.shape[1], 1)
    new_patient_data_reshapedH = new_patient_data_scaledH.reshape(1, new_patient_data_scaledH.shape[1], 1)
    new_patient_data_reshapedI = new_patient_data_scaledI.reshape(1, new_patient_data_scaledI.shape[1], 1)
    new_patient_data_reshapedJ = new_patient_data_scaledJ.reshape(1, new_patient_data_scaledJ.shape[1], 1)

    #Analyze if the patient has risk of CVD with CNN model
    risk_disease_cnn = risk_cnn(cnn_model, new_patient_data_reshaped)
    print(f"CNN, Disease risk A: {risk_disease_cnn}")
    risk_disease_cnnB = risk_cnn(cnn_model, new_patient_data_reshapedB)
    print(f"CNN, Disease risk B: {risk_disease_cnnB}")
    risk_disease_cnnC = risk_cnn(cnn_model, new_patient_data_reshapedC)
    print(f"CNN, Disease risk C: {risk_disease_cnnC}")
    risk_disease_cnnD = risk_cnn(cnn_model, new_patient_data_reshapedD)
    print(f"CNN, Disease risk D: {risk_disease_cnnD}")
    risk_disease_cnnE = risk_cnn(cnn_model, new_patient_data_reshapedE)
    print(f"CNN, Disease risk E: {risk_disease_cnnE}")
    risk_disease_cnnF = risk_cnn(cnn_model, new_patient_data_reshapedF)
    print(f"CNN, Disease risk F: {risk_disease_cnnF}")
    risk_disease_cnnG = risk_cnn(cnn_model, new_patient_data_reshapedG)
    print(f"CNN, Disease risk G: {risk_disease_cnnG}")
    risk_disease_cnnH = risk_cnn(cnn_model, new_patient_data_reshapedH)
    print(f"CNN, Disease risk H: {risk_disease_cnnH}")
    risk_disease_cnnI = risk_cnn(cnn_model, new_patient_data_reshapedI)
    print(f"CNN, Disease risk I: {risk_disease_cnnI}")
    risk_disease_cnnJ = risk_cnn(cnn_model, new_patient_data_reshapedJ)
    print(f"CNN, Disease risk J: {risk_disease_cnnJ}")

    #Analyze probability of disease with CNN model
    probabilities_cnn = prob_cnn(cnn_model, new_patient_data_reshaped)
    print(f"CNN, Probability of having cardiovascular disease A: {probabilities_cnn}")
    probabilities_cnnB = prob_cnn(cnn_model, new_patient_data_reshapedB)
    print(f"CNN, Probability of having cardiovascular disease B: {probabilities_cnnB}")
    probabilities_cnnC = prob_cnn(cnn_model, new_patient_data_reshapedC)
    print(f"CNN, Probability of having cardiovascular disease C: {probabilities_cnnC}")
    probabilities_cnnD = prob_cnn(cnn_model, new_patient_data_reshapedD)
    print(f"CNN, Probability of having cardiovascular disease D: {probabilities_cnnD}")
    probabilities_cnnE = prob_cnn(cnn_model, new_patient_data_reshapedE)
    print(f"CNN, Probability of having cardiovascular disease E: {probabilities_cnnE}")
    probabilities_cnnF = prob_cnn(cnn_model, new_patient_data_reshapedF)
    print(f"CNN, Probability of having cardiovascular disease F: {probabilities_cnnF}")
    probabilities_cnnG = prob_cnn(cnn_model, new_patient_data_reshapedG)
    print(f"CNN, Probability of having cardiovascular disease G: {probabilities_cnnG}")
    probabilities_cnnH = prob_cnn(cnn_model, new_patient_data_reshapedH)
    print(f"CNN, Probability of having cardiovascular disease H: {probabilities_cnnH}")
    probabilities_cnnI = prob_cnn(cnn_model, new_patient_data_reshapedI)
    print(f"CNN, Probability of having cardiovascular disease I: {probabilities_cnnI}")
    probabilities_cnnJ = prob_cnn(cnn_model, new_patient_data_reshapedJ)
    print(f"CNN, Probability of having cardiovascular disease J: {probabilities_cnnJ}")

    #Analyze feature importance for CNN model
    variables_cnn(cnn_model, X_test_reshaped, X_train_scaled)

    #Analyze accuracy of CNN model
    cnn_model_accuracy = accuracy_cnn(cnn_model, X_test_reshaped, y_test)
    print(f"CNN Accuracy on the test set: {cnn_model_accuracy}")

    #Check accuracy of the validation set predictions
    valid_cnn = validation_cnn(cnn_model, X_validation_reshaped, y_validation)
    valid_cnn2 = validation_cnn(cnn_model, X_validation_reshaped2, y_validation2)
    valid_cnn3 = validation_cnn(cnn_model, X_validation_reshaped3, y_validation3)

    print(f"CNN, Accuracy of the validation set 1 predictions: {valid_cnn:.4f}")
    print(f"CNN, Accuracy of the validation set 2 predictions: {valid_cnn2:.4f}")
    print(f"CNN, Accuracy of the validation set 3 predictions: {valid_cnn3:.4f}")

#ANN
def train_ann(X_train, y_train):
    #Create and train the MLPClassifier
    ann_classifier = MLPClassifier(hidden_layer_sizes=(64,64), activation='relu', solver='adam', random_state=42, max_iter=30)
    ann_classifier.fit(X_train, y_train)
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
    new_patient_data_scaled = new_patient_data
    #Make predictions on the new patient
    prediction = ann_classifier.predict(new_patient_data_scaled)
    return prediction[0]

def prob_ann(ann_classifier, new_patient_data):
    new_patient_data_scaled = new_patient_data
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
        ann_model = load_ann('ann_model_70000.joblib')
    else:
        #Train ANN model
        ann_model = train_ann(X_train, y_train)

        #Save ANN model
        save_ann(ann_model, 'ann_model.joblib')
    return ann_model

def main_ann():

    ann_model = ANNmodel(False)

    #New patient data in numpy array
    new_patient_data = np.array([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
    new_patient_dataB = np.array([[ageB, genderB, heightB, weightB, ap_hiB, ap_loB, cholesterolB, glucB, smokeB, alcoB, activeB]])
    new_patient_dataC = np.array([[ageC, genderC, heightC, weightC, ap_hiC, ap_loC, cholesterolC, glucC, smokeC, alcoC, activeC]])
    new_patient_dataD = np.array([[ageD, genderD, heightD, weightD, ap_hiD, ap_loD, cholesterolD, glucD, smokeD, alcoD, activeD]])
    new_patient_dataE = np.array([[ageE, genderE, heightE, weightE, ap_hiE, ap_loE, cholesterolE, glucE, smokeE, alcoE, activeE]])
    new_patient_dataF = np.array([[ageF, genderF, heightF, weightF, ap_hiF, ap_loF, cholesterolF, glucF, smokeF, alcoF, activeF]])
    new_patient_dataG = np.array([[ageG, genderG, heightG, weightG, ap_hiG, ap_loG, cholesterolG, glucG, smokeG, alcoG, activeG]])
    new_patient_dataH = np.array([[ageH, genderH, heightH, weightH, ap_hiH, ap_loH, cholesterolH, glucH, smokeH, alcoH, activeH]])
    new_patient_dataI = np.array([[ageI, genderI, heightI, weightI, ap_hiI, ap_loI, cholesterolI, glucI, smokeI, alcoI, activeI]])
    new_patient_dataJ = np.array([[ageJ, genderJ, heightJ, weightJ, ap_hiJ, ap_loJ, cholesterolJ, glucJ, smokeJ, alcoJ, activeJ]])
    
    #Analyze if the patient has risk of CVD with ANN model
    risk_disease_ann = risk_ann(ann_model, new_patient_data)
    print(f"ANN, Disease risk A: {risk_disease_ann}")
    risk_disease_annB = risk_ann(ann_model, new_patient_dataB)
    print(f"ANN, Disease risk B: {risk_disease_annB}")
    risk_disease_annC = risk_ann(ann_model, new_patient_dataC)
    print(f"ANN, Disease risk C: {risk_disease_annC}")
    risk_disease_annD = risk_ann(ann_model, new_patient_dataD)
    print(f"ANN, Disease risk D: {risk_disease_annD}")
    risk_disease_annE = risk_ann(ann_model, new_patient_dataE)
    print(f"ANN, Disease risk E: {risk_disease_annE}")
    risk_disease_annF = risk_ann(ann_model, new_patient_dataF)
    print(f"ANN, Disease risk F: {risk_disease_annF}")
    risk_disease_annG = risk_ann(ann_model, new_patient_dataG)
    print(f"ANN, Disease risk G: {risk_disease_annG}")
    risk_disease_annH = risk_ann(ann_model, new_patient_dataH)
    print(f"ANN, Disease risk H: {risk_disease_annH}")
    risk_disease_annI = risk_ann(ann_model, new_patient_dataI)
    print(f"ANN, Disease risk I: {risk_disease_annI}")
    risk_disease_annJ = risk_ann(ann_model, new_patient_dataJ)
    print(f"ANN, Disease risk J: {risk_disease_annJ}")

    #Analyze probability of disease with ANN model
    probabilities_ann = prob_ann(ann_model, new_patient_data)
    print(f"ANN, Probability of having cardiovascular disease A: {probabilities_ann}")
    probabilities_annB = prob_ann(ann_model, new_patient_dataB)
    print(f"ANN, Probability of having cardiovascular disease B: {probabilities_annB}")
    probabilities_annC = prob_ann(ann_model, new_patient_dataC)
    print(f"ANN, Probability of having cardiovascular disease C: {probabilities_annC}")
    probabilities_annD = prob_ann(ann_model, new_patient_dataD)
    print(f"ANN, Probability of having cardiovascular disease D: {probabilities_annD}")
    probabilities_annE = prob_ann(ann_model, new_patient_dataE)
    print(f"ANN, Probability of having cardiovascular disease E: {probabilities_annE}")
    probabilities_annF = prob_ann(ann_model, new_patient_dataF)
    print(f"ANN, Probability of having cardiovascular disease F: {probabilities_annF}")
    probabilities_annG = prob_ann(ann_model, new_patient_dataG)
    print(f"ANN, Probability of having cardiovascular disease G: {probabilities_annG}")
    probabilities_annH = prob_ann(ann_model, new_patient_dataH)
    print(f"ANN, Probability of having cardiovascular disease H: {probabilities_annH}")
    probabilities_annI = prob_ann(ann_model, new_patient_dataI)
    print(f"ANN, Probability of having cardiovascular disease I: {probabilities_annI}")
    probabilities_annJ = prob_ann(ann_model, new_patient_dataJ)
    print(f"ANN, Probability of having cardiovascular disease J: {probabilities_annJ}")

    #Analyze feature importance for ANN model
    variables_ann(ann_model, X_train)

    #Analyze accuracy of ANN model
    ann_model_accuracy = accuracy_ann(ann_model, X_test, y_test)
    print(f"ANN Accuracy on the test set: {ann_model_accuracy}")

    #Check accuracy of the validation set predictions
    valid_ann = validation_ann(ann_model, X_validation, y_validation)
    valid_ann2 = validation_ann(ann_model, X_validation2, y_validation2)
    valid_ann3 = validation_ann(ann_model, X_validation3, y_validation3)

    print(f"ANN, Accuracy of the validation set 1 predictions: {valid_ann:.4f}")
    print(f"ANN, Accuracy of the validation set 2 predictions: {valid_ann2:.4f}")
    print(f"ANN, Accuracy of the validation set 3 predictions: {valid_ann3:.4f}")

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

ageE = 60
genderE = 1
heightE = 157
weightE = 74
ap_hiE = 130
ap_loE = 80
cholesterolE = 1
glucE = 1
smokeE = 0
alcoE = 0
activeE = 1

ageF = 50
genderF = 2
heightF = 174
weightF = 74
ap_hiF = 110
ap_loF = 70
cholesterolF = 2
glucF = 2
smokeF = 0  
alcoF = 0  
activeF = 1

ageG = 60
genderG = 1
heightG = 160
weightG = 59
ap_hiG = 110
ap_loG = 70
cholesterolG = 1
glucG = 1
smokeG = 0  
alcoG = 0  
activeG = 1

ageH = 46
genderH = 1
heightH = 170
weightH = 68
ap_hiH = 120
ap_loH = 80
cholesterolH = 1
glucH = 1
smokeH = 1  
alcoH = 0  
activeH = 1

ageI = 49
genderI = 2
heightI = 176
weightI = 109
ap_hiI = 120
ap_loI = 80
cholesterolI = 2
glucI = 1
smokeI = 0  
alcoI = 0  
activeI = 1

ageJ = 61
genderJ = 1
heightJ = 158
weightJ = 126
ap_hiJ = 140
ap_loJ = 90
cholesterolJ = 2
glucJ = 2
smokeJ = 0  
alcoJ = 0  
activeJ = 1

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

def diagnose_patientE():

    new_patient = pd.DataFrame({
        'age': [60],
        'gender': [1],
        'height': [157],
        'weight': [74],
        'ap_hi': [130],
        'ap_lo': [80],
        'cholesterol': [1],
        'gluc': [1],
        'smoke': [0],
        'alco': [0],
        'active': [1]
    })

    return new_patient

def diagnose_patientF():

    new_patient = pd.DataFrame({
        'age': [50],
        'gender': [2],
        'height': [174],
        'weight': [74],
        'ap_hi': [110],
        'ap_lo': [70],
        'cholesterol': [2],
        'gluc': [2],
        'smoke': [0],
        'alco': [0],
        'active': [1]
    })

    return new_patient

def diagnose_patientG():

    new_patient = pd.DataFrame({
        'age': [60],
        'gender': [1],
        'height': [160],
        'weight': [59],
        'ap_hi': [110],
        'ap_lo': [70],
        'cholesterol': [1],
        'gluc': [1],
        'smoke': [0],
        'alco': [0],
        'active': [1]
    })

    return new_patient

def diagnose_patientH():

    new_patient = pd.DataFrame({
        'age': [46],
        'gender': [1],
        'height': [170],
        'weight': [68],
        'ap_hi': [120],
        'ap_lo': [80],
        'cholesterol': [1],
        'gluc': [1],
        'smoke': [1],
        'alco': [0],
        'active': [1]
    })

    return new_patient

def diagnose_patientI():

    new_patient = pd.DataFrame({
        'age': [49],
        'gender': [2],
        'height': [176],
        'weight': [109],
        'ap_hi': [120],
        'ap_lo': [80],
        'cholesterol': [2],
        'gluc': [1],
        'smoke': [0],
        'alco': [0],
        'active': [1]
    })

    return new_patient

def diagnose_patientJ():

    new_patient = pd.DataFrame({
        'age': [61],
        'gender': [1],
        'height': [158],
        'weight': [126],
        'ap_hi': [140],
        'ap_lo': [90],
        'cholesterol': [2],
        'gluc': [2],
        'smoke': [0],
        'alco': [0],
        'active': [1]
    })

    return new_patient


if __name__ == "__main__":

    #Load CSV file
    data = pd.read_csv(r"C:\Users\mario\OneDrive\Escritorio\Uni\SE\SKOVDE\A3\ThesisMethods\cardio_train.csv")

    #Data preprocessing
    data = data[(data['height'] >= 100) & (data['height'] <= 210) & (data['weight'] >= 20) & (data['weight'] <= 250)]
    data = data.dropna()
    data_tail = data.tail(6000)
    data_val = data_tail.sample(2000)
    data_val2 = data_tail.sample(2000)
    data_val3 = data_tail.sample(2000)
    #data = data.head(5000)
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

    X_train_scaled = scaler.fit_transform(X_train)

    X_validation_scaled = scaler.transform(X_validation)
    X_validation_scaled2 = scaler.transform(X_validation2)
    X_validation_scaled3 = scaler.transform(X_validation3)

    X_test_scaled = scaler.transform(X_test)

    #Reshape data for 1D CNN
    X_validation_reshaped = X_validation_scaled.reshape(X_validation_scaled.shape[0], X_validation_scaled.shape[1], 1)
    X_validation_reshaped2 = X_validation_scaled2.reshape(X_validation_scaled2.shape[0], X_validation_scaled2.shape[1], 1)
    X_validation_reshaped3 = X_validation_scaled3.reshape(X_validation_scaled3.shape[0], X_validation_scaled3.shape[1], 1)

    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    main_ann()
