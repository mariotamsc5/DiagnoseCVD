# DiagnoseCVD
## Introduction
In this project four AI models were used: Decision Tree (DT), Random Forest (RF), Convolutional Neural Network (CNN), and Artificial Neural Network (ANN). The tasks based on a file containing a large amount of patient data are to determine the importance of some variables in the diagnosis of a cardiovascular disease (CVD) and the accuracy of the trained models. Also, three validation groups of patients were used to test the trained models.The tasks based on the data of a patient are the diagnosis and the probability of having a CVD.
The models were trained with 4.000, 12.000, 28.000, 40.000 and 56.000 patients.
### User interface
The goal of the user interface is to ease the diagnosis of a CVD in a patient based on its data on specific variables. It offers a tool to analyse a csv file and give out the accuracy of the model and the bar graph with the importance of the variables to diagnose a CVD.
The interface is divided into two parts. The first part is where an individual patient data is used to carry out the diagnosis of a CVD, and the second part is to train and test the models with a csv file containing a large group of patients' data.

In both parts it is possible to select the model to use among DT, RF, CNN, and ANN. In addition, in the first part the user can chose from loading a pretrained model or training it again. The output of the diagnosis of an individual patient will give a positive or negative result to said CVD diagnosis, and the probability of being affected by the same type of disease, as a number ranging from 0.00 to 1.00. For the csv file analysis part, the output will be the bar graph of the variables' importance and the accuracy value of the model trained with data set given.
## Contents
* README.md

File with all the information about the documents and scripts contained in the repository, including the instructions on how to work with the script files.
* CVD_project
    - bar_graphs
      
    Folder with the variables' importance bar graphs resulted from training the four models with the five different quantities of patients.
    - data
      
    Cardiovascular dataset utilized to train, test and validate various AI models.
    The dataset is a comma delimited (csv) file retrieved from Kaggle Cardiovascular Disease dataset, containing publicly available information on features associated with cardiovascular issues (Ulianova, 2019).
    - project_scripts

    **CODE_MariaSaezCarazo.py**: Python script with all the tasks analysed in the project.
  
    **USER_INTER_MariaSaezCarazo.py**: Python script for the user interface.
    - trained_models

    Folder with the trained models with the five different quantities of patients. The files starting with 5000 were trained with 4.000 patients, the ones starting with 15000 were trained with 12.000 patients, the ones starting with 35000 were trained with 28.000 patients, the ones starting with 50000 were trained with 40.000 patients, and the ones starting with 70000 were trained with 56.000 patients. The files without a number were trained using 80% of the patients in the file, which in the case of the data file used for the project is equal to 56.000 patients used for training.
## Use
* **CODE_MariaSaezCarazo.py**
    - Required packages: pandas, numpy,sklearn, joblib,matplotlib, tensorflow and keras.
    - Required input: file in data folder, or other with the same characteristics and variables of each patient.
    - Output: bar graph with variables' importance for a CVD diagnosis, diagnosis of four patients (0=healthy, 1=disease), probability of the same four patients of having a CVD, accuracy of the model on the test set, and accuracy of the validation sets diagnosis.
      
Once the script, data and trained models files have been downloaded, the following changes could be made to the script file based on the desired output.
 
    - line 77, 85, 222, 229, 379, 386, 550, 556: to change the name of the model file. Eg. load_dt('dt_model.joblib'), save_ann(ann_model, 'ann_model.joblib')
    - line 90, 234, 391, 561: to load a pretrained model or train a new one. Eg. DTmodel(True), RFmodel(False)
    - from line 636 to 932: to change the data from the patients to diagnose.
    - line 941: to load the correct (csv) file. Eg. pd.read_csv(r"C:...\cardio_train.csv")
    - line 950: to train and test with the desired quantitiy of patients. Eg. 5000, 15000, 35000, 50000 or comment the line
    - line 986: to work with the desired model. Eg. main_dt(), main_rf(), main_cnn(), or main_ann()
Finally, to run the file correctly the following command has to be used in the terminal in the appropriate directory:
```
python .\CODE_MariaSaezCarazo.py 
```

* **USER_INTER_MariaSaezCarazo.py**
    - Required packages: pandas, numpy,sklearn, joblib,matplotlib, tensorflow, keras, tkinter and customtkinter
    - Required input: file in data folder, or other with the same characteristics and variables of each patient, and data of the patient to diagnose.
    - Output: diagnosis of the patient, probability of the same patient of having a CVD, bar graph with variables' importance for a CVD diagnosis, accuracy of the model on the test set.
  
Once the script, data and trained models files have been downloaded, the following changes could be made to the script file.

    - line 52, 72, 106, 127, 170, 194, 231, 254: to change the name of the model file. Eg. load_dt('dt_model.joblib'), save_ann(ann_model, 'ann_model.joblib')
    - line 56, 110, 173, 234: to load the correct (csv) file. Eg. pd.read_csv(r"C:...\cardio_train.csv")
Finally, to get the user interface correctly the following command has to be used in the terminal in the appropriate directory:
```
python .\USER_INTER_MariaSaezCarazo.py 
```
