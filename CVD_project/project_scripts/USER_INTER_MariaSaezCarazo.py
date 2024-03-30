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
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tkinter import *
import customtkinter
from customtkinter import filedialog
from CTkMessagebox import CTkMessagebox

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
    #Load saved model
    return joblib.load(dt_filename)

def predict_risk_dt(dt_classifier, new_patient):
    #Make predictions on the new patient
    prediction = dt_classifier.predict(new_patient)
    #Interpret the prediction
    predicted_class = "Positive" if prediction[0] == 1 else "Negative"
    return predicted_class

def predict_proba_dt(dt_classifier, new_patient):
    #Make predictions on the new patient
    return dt_classifier.predict_proba(new_patient)

def DTmodel(exists):
    if exists:
        dt_model = load_dt('dt_model.joblib')

    else:
        #Load CSV file
        data = pd.read_csv(r"C:\Users\mario\OneDrive\Escritorio\Uni\SE\SKOVDE\A3\ThesisMethods\cardio_train.csv")
        
        #Data preprocessing
        data = data[(data['height'] >= 100) & (data['height'] <= 210) & (data['weight'] >= 20) & (data['weight'] <= 250)]        
        data = data.dropna()
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

def predict_risk_rf(rf_classifier, new_patient):
    #Make predictions on the new patient
    prediction = rf_classifier.predict(new_patient)
    #Interpret the prediction
    predicted_class = "Positive" if prediction[0] == 1 else "Negative"
    return predicted_class

def predict_proba_rf(rf_classifier, new_patient):
    #Make predictions on the new patient
    probability = rf_classifier.predict_proba(new_patient)
    return probability[0][1]

def RFmodel(exists):
    if exists:
        rf_model = load_rf('rf_model.joblib')

    else:
        #Load CSV file
        data = pd.read_csv(r"C:\Users\mario\OneDrive\Escritorio\Uni\SE\SKOVDE\A3\ThesisMethods\cardio_train.csv")
        data = data.drop('id', axis=1)
        
        #Data preprocessing
        data = data[(data['height'] >= 100) & (data['height'] <= 210) & (data['weight'] >= 20) & (data['weight'] <= 250)]        
        data = data.dropna()
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

def risk_cnn(cnn_classifier, new_patient_data_reshaped):
    #Make predictions on the new patient
    prediction_prob = cnn_classifier.predict(new_patient_data_reshaped)
    #Interpret the prediction
    predicted_class = "Positive" if prediction_prob[0][0] >= 0.5 else "Negative"
    return predicted_class

def prob_cnn(cnn_classifier,new_patient_data_reshaped):
    #Make predictions on the new patient
    prediction_prob = cnn_classifier.predict(new_patient_data_reshaped)
    #Prediction show as 0.00
    formatted_prediction = "{:.2f}".format(prediction_prob[0][0])
    return formatted_prediction

def CNNmodel(exists):
    if exists:
        cnn_model = load_cnn('cnn_model.h5')
    else:
        #Load CSV file
        data = pd.read_csv(r"C:\Users\mario\OneDrive\Escritorio\Uni\SE\SKOVDE\A3\ThesisMethods\cardio_train.csv")
        
        #Data preprocessing
        data = data[(data['height'] >= 100) & (data['height'] <= 210) & (data['weight'] >= 20) & (data['weight'] <= 250)]        
        data = data.dropna()
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


#ANN
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

def risk_ann(ann_classifier, new_patient_data):
    new_patient_data_scaled = scaler.fit_transform(new_patient_data)
    #Make predictions on the new patient
    prediction = ann_classifier.predict(new_patient_data_scaled)
    predicted_class = "Positive" if prediction[0] == 1 else "Negative"
    return predicted_class

def prob_ann(ann_classifier, new_patient_data):
    new_patient_data_scaled = scaler.transform(new_patient_data)
    #Make predictions on the new patient
    probability = ann_classifier.predict_proba(new_patient_data_scaled)[:, 1]
    #Probability to show as 0.00
    formatted_probability = "{:.2f}".format(probability[0])
    return formatted_probability

def ANNmodel(exists):
    if exists:
        ann_model = load_ann('ann_model.joblib')
    else:
        #Load CSV file
        data = pd.read_csv(r"C:\Users\mario\OneDrive\Escritorio\Uni\SE\SKOVDE\A3\ThesisMethods\cardio_train.csv")
        
        # Data preprocessing
        data = data[(data['height'] >= 100) & (data['height'] <= 210) & (data['weight'] >= 20) & (data['weight'] <= 250)]        
        data = data.dropna()
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


#Function to get user input and perform diagnosis
def diagnose_patient():
    age = int(age_entry.get())
    gender = int(gender_entry.get())
    height = int(height_entry.get())
    weight = int(weight_entry.get())
    ap_hi = int(ap_hi_entry.get())
    ap_lo = int(ap_lo_entry.get())
    cholesterol = int(chol_entry.get())
    gluc = int(gluc_entry.get())
    smoke = int(smoke_entry.get())
    alco = int(alco_entry.get())
    active = int(active_entry.get())

    new_patient = np.array([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
    return new_patient


#Function to handle model selection and prediction
def analyze_patient():
    scaler = StandardScaler()
    model_choice = model_options_pat.get()
    exists = True if model_exists_var.get() == 1 else False
    new_patient = diagnose_patient()

    if model_choice == "Decision Tree":
        dt_model = DTmodel(exists)
        risk_disease_dt = predict_risk_dt(dt_model, new_patient)
        probabilities_dt = predict_proba_dt(dt_model, new_patient)
        CTkMessagebox(title="Result", message=f"DT, Diagnosis for CVD: {risk_disease_dt}\nDT, Probability of having a CVD: {probabilities_dt}")

    elif model_choice == "Random Forest":
        rf_model = RFmodel(exists)
        risk_disease_rf = predict_risk_rf(rf_model, new_patient)
        probabilities_rf = predict_proba_rf(rf_model, new_patient)
        CTkMessagebox(title="Result", message=f"RF, Diagnosis for CVD: {risk_disease_rf}\nRF, Probability of having a CVD: {probabilities_rf}")

    elif model_choice == "CNN":
        cnn_model = CNNmodel(exists)
        new_patient_data_scaled = scaler.fit_transform(new_patient)
        new_patient_data_reshaped = new_patient_data_scaled.reshape(1, new_patient_data_scaled.shape[1], 1)
        risk_disease_cnn = risk_cnn(cnn_model, new_patient_data_reshaped)
        probabilities_cnn = prob_cnn(cnn_model, new_patient_data_reshaped)
        CTkMessagebox(title="Result", message=f"CNN, Diagnosis for CVD: {risk_disease_cnn}\nCNN, Probability of having a CVD: {probabilities_cnn}")

    elif model_choice == "ANN":
        ann_model = ANNmodel(exists)
        risk_disease_ann = risk_ann(ann_model, new_patient)
        probabilities_ann = prob_ann(ann_model, new_patient)
        CTkMessagebox(title="Result", message=f"ANN, Diagnosis for CVD: {risk_disease_ann}\nANN, Probability of having a CVD: {probabilities_ann}")

#Function to load csv file with data
def load_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    try:
        root.data = pd.read_csv(file_path)
        CTkMessagebox(title="Success", message="CSV file loaded successfully!")
    except Exception as e:
        CTkMessagebox(title="Error", message=str(e))

#Analyze data based on the method selected
def analyze_data():
    try:
            
        method = model_options.get()

        if method == "Decision Tree":
            run_decision_tree()
        elif method == "Random Forest":
            run_random_forest()
        elif method == "CNN":
            run_cnn()
        elif method == "ANN":
            run_ann()
        else:
            CTkMessagebox(title="Error", message="Please select a method of analysis.")

    except Exception as e:
        CTkMessagebox(title="Error", message=str(e))

def run_decision_tree():
    #Preprocess and split data
    data = root.data.drop(['id'], axis=1).dropna()
    data = data[(data['height'] >= 100) & (data['height'] <= 210) & (data['weight'] >= 20) & (data['weight'] <= 250)]
    X = data.drop('cardio', axis=1)
    y = data['cardio']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Train model
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    
    #Calculate accuracy
    dt_model_accuracy = accuracy_score(y_test, dt_classifier.predict(X_test))

    #Plot feature importance
    importances = dt_classifier.feature_importances_
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    
    #Create a separate window for the accuracy
    accuracy_window = customtkinter.CTkToplevel(root)
    accuracy_window.title("Accuracy")
    accuracy_window.geometry("200x100")

    accuracy_label = customtkinter.CTkLabel(accuracy_window, text="Accuracy on the test set:")
    accuracy_label.pack()

    accuracy_value = customtkinter.CTkLabel(accuracy_window, text=f"{dt_model_accuracy:.2f}")
    accuracy_value.pack()
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='lightgreen')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance for CVD Diagnosis with Decision Tree')
    plt.show()



def run_random_forest():
    #Preprocess and split data
    data = root.data.drop(['id'], axis=1).dropna()
    data = data[(data['height'] >= 100) & (data['height'] <= 210) & (data['weight'] >= 20) & (data['weight'] <= 250)]
    X = data.drop('cardio', axis=1)
    y = data['cardio']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Train model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    #Calculate accuracy
    rf_model_accuracy = accuracy_score(y_test, rf_classifier.predict(X_test))

    #Plot feature importance
    importances = rf_classifier.feature_importances_
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    
    #Create a separate window for the accuracy
    accuracy_window = customtkinter.CTkToplevel(root)
    accuracy_window.title("Accuracy")
    accuracy_window.geometry("200x100")

    accuracy_label = customtkinter.CTkLabel(accuracy_window, text="Accuracy on the test set:")
    accuracy_label.pack()

    accuracy_value = customtkinter.CTkLabel(accuracy_window, text=f"{rf_model_accuracy:.2f}")
    accuracy_value.pack()

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='lightgreen')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance for CVD Diagnosis with Random Forest')
    plt.show()


def run_cnn():
    #Preprocess and split data
    data = root.data.drop(['id'], axis=1).dropna()
    data = data[(data['height'] >= 100) & (data['height'] <= 210) & (data['weight'] >= 20) & (data['weight'] <= 250)]
    X = data.drop('cardio', axis=1)
    y = data['cardio']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Standardize input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Reshape input for CNN
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    cnn_classifier = Sequential()
    cnn_classifier.add(Conv1D(filters=X_train_reshaped.shape[1], kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
    cnn_classifier.add(MaxPooling1D(pool_size=2))
    cnn_classifier.add(Flatten())
    cnn_classifier.add(Dense(64, activation='relu'))
    cnn_classifier.add(Dense(1, activation='sigmoid'))

    cnn_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #Train model
    cnn_classifier.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test))

    #Calculate accuracy
    cnn_model_accuracy = cnn_classifier.evaluate(X_test_reshaped, y_test)[1]

    #Plot feature importance
    layer_output = Model(inputs=cnn_classifier.input, outputs=cnn_classifier.layers[0].output)
    filters = layer_output.predict(X_test_reshaped)
    mean_abs_values = np.mean(np.abs(filters), axis=(0, 1))
    sorted_indices = np.argsort(mean_abs_values)[::-1]
    sorted_mean_abs_values = mean_abs_values[sorted_indices]
    
    #Create a separate window for the accuracy
    accuracy_window = customtkinter.CTkToplevel(root)
    accuracy_window.title("Accuracy")
    accuracy_window.geometry("200x100")

    accuracy_label = customtkinter.CTkLabel(accuracy_window, text="Accuracy on the test set:")
    accuracy_label.pack()

    accuracy_value = customtkinter.CTkLabel(accuracy_window, text=f"{cnn_model_accuracy:.2f}")
    accuracy_value.pack()
    
    plt.barh(range(len(sorted_mean_abs_values)), sorted_mean_abs_values, color='lightgreen')
    plt.yticks(range(X_train_scaled.shape[1]), [X.columns[i] for i in sorted_indices])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance for CVD Diagnosis with CNN')
    plt.show()

def run_ann():
    #Preprocess and split data
    data = root.data.drop(['id'], axis=1).dropna()
    data = data[(data['height'] >= 100) & (data['height'] <= 210) & (data['weight'] >= 20) & (data['weight'] <= 250)]
    X = data.drop('cardio', axis=1)
    y = data['cardio']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Standardize input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Train model
    ann_classifier = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam', random_state=42, max_iter=1000, tol=1e-4)
    ann_classifier.fit(X_train_scaled, y_train)

    #Calculate accuracy
    ann_model_accuracy = accuracy_score(y_test, ann_classifier.predict(X_test_scaled))

    #Plot feature importance
    weights_input_hidden = ann_classifier.coefs_[0].T
    feature_importance = np.abs(weights_input_hidden).mean(axis=0)
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_feature_importance = feature_importance[sorted_indices]
    sorted_feature_names = X.columns[sorted_indices]
    
    #Create a separate window for the accuracy
    accuracy_window = customtkinter.CTkToplevel(root)
    accuracy_window.title("Accuracy")
    accuracy_window.geometry("200x100")

    accuracy_label = customtkinter.CTkLabel(accuracy_window, text="Accuracy on the test set:")
    accuracy_label.pack()

    accuracy_value = customtkinter.CTkLabel(accuracy_window, text=f"{ann_model_accuracy:.2f}")
    accuracy_value.pack()
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(X_train_scaled.shape[1]), sorted_feature_importance, color='lightgreen')
    plt.yticks(range(X_train_scaled.shape[1]), sorted_feature_names)
    plt.xlabel('Mean Absolute Weight')
    plt.title('Feature Importance in Artificial Neural Network (ANN)')
    plt.show()

#Set appearance of the interface
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

#Create main window
root = customtkinter.CTk()
root.title("CVD Diagnosis")
root.geometry("800x700")

#Add two different tabs: for patient and for file
my_tabs = customtkinter.CTkTabview(root, width=700, height=600, corner_radius=20)
my_tabs.pack(pady=10)
tab_pat = my_tabs.add("1 patient")
tab_grp = my_tabs.add("csv file")

#Frame for the selection of methods
frame_method = customtkinter.CTkFrame(tab_pat, width=200, height=400, fg_color="#6a72a3", border_color="#8673a1", border_width=2)
frame_method.place(x=70, y=20)

#Model Selection
#Label for the selection of methods
label_method = customtkinter.CTkLabel(frame_method, text="Select method for the analysis")
label_method.place(x=10, y=20)

#Option menu for methods
models = ["Decision Tree", "Random Forest", "CNN","ANN"]
model_options_pat = customtkinter.CTkOptionMenu(frame_method, values=models)
model_options_pat.place(x=30, y=70)

#Checkbox to load pretrained model
model_exists_var = customtkinter.IntVar()
model_exists_checkbox = customtkinter.CTkCheckBox(frame_method, text="Load Pretrained Model", variable=model_exists_var)
model_exists_checkbox.place(x=20, y=200)

#Frame for the input of patient data
frame_data = customtkinter.CTkFrame(tab_pat, width=200, height=400, fg_color="#6a72a3", border_color="#8673a1", border_width=2)
frame_data.place(x=370, y=20)

#Label for the introduction of patient data
label_method = customtkinter.CTkLabel(frame_data, text="Introduce patient data")
label_method.place(x=30, y=20)

#Entries for the patient data
age_entry = customtkinter.CTkEntry(frame_data, placeholder_text="Age(years)")
age_entry.place(x=30, y=50)

gender_entry = customtkinter.CTkEntry(frame_data, placeholder_text="1:woman/2:men")
gender_entry.place(x=30, y=80)

height_entry = customtkinter.CTkEntry(frame_data, placeholder_text="Height(cm)")
height_entry.place(x=30, y=110)

weight_entry = customtkinter.CTkEntry(frame_data, placeholder_text="Weight(kg)")
weight_entry.place(x=30, y=140)

ap_hi_entry = customtkinter.CTkEntry(frame_data, placeholder_text="Systolic pressure")
ap_hi_entry.place(x=30, y=170)

ap_lo_entry = customtkinter.CTkEntry(frame_data, placeholder_text="Diastolic pressure")
ap_lo_entry.place(x=30, y=200)

chol_entry = customtkinter.CTkEntry(frame_data, placeholder_text="Cholesterol(1-3)")
chol_entry.place(x=30, y=230)

gluc_entry = customtkinter.CTkEntry(frame_data, placeholder_text="Glucose(1-3)")
gluc_entry.place(x=30, y=260)

smoke_entry = customtkinter.CTkEntry(frame_data, placeholder_text="Smoker(0:no/1:yes)")
smoke_entry.place(x=30, y=290)

alco_entry = customtkinter.CTkEntry(frame_data, placeholder_text="Alc. use(0:no/1:yes)")
alco_entry.place(x=30, y=320)

active_entry = customtkinter.CTkEntry(frame_data, placeholder_text="PA(0:no/1:yes)")
active_entry.place(x=30, y=350)

#Diagnosis Button
scaler = StandardScaler()
search_button = customtkinter.CTkButton(tab_pat, text="Diagnose", command=analyze_patient)
search_button.place(x=250, y=460)


#Frame for the selection of methods in the file tab
frame_method2 = customtkinter.CTkFrame(tab_grp, width=200, height=400, fg_color="#6a72a3", border_color="#8673a1", border_width=2)
frame_method2.place(x=70, y=20)

#Model Selection
#Label for the selection of methods
label_method = customtkinter.CTkLabel(frame_method2, text="Select method for the analysis")
label_method.place(x=10, y=20)

#Option menu for methods
models = ["Decision Tree", "Random Forest", "CNN","ANN"]
model_options = customtkinter.CTkOptionMenu(frame_method2, values=models)
model_options.place(x=30, y=70)

#Frame for the selection of the file
frame_file = customtkinter.CTkFrame(tab_grp, width=200, height=200, fg_color="#6a72a3", border_color="#8673a1", border_width=2)
frame_file.place(x=370, y=130)

#Label for the selection of methods
load_data_button = customtkinter.CTkButton(frame_file, text="Load CSV", command=load_csv)
load_data_button.place(x=30, y=70)

#Analyze button
search_button2 = customtkinter.CTkButton(tab_grp, text="Analyze", command=analyze_data)
search_button2.place(x=250, y=460)


root.mainloop()