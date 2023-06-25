# Importing the necessary libraries.
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import logging

# Create a folder for log files if it doesn't exist
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)
# Specify the path of the data folder
data_folder = os.path.join(os.getcwd(), "code", "data")

# Configure logging for Data_preprocessing_part_2.log
log_file_part_4 = os.path.join(log_folder, "Model_selection.log")
logging.basicConfig(filename=log_file_part_4, level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")

logging.info("Reading training dataset.........")
file_path = os.path.join(data_folder, "x_train.csv")
x_train =  pd.read_csv(file_path)
file_path = os.path.join(data_folder, "y_train.csv")
y_train =  pd.read_csv(file_path).values.ravel()
logging.info("Read the training data successfully.")

logging.info("Reading testing dataset........")
file_path = os.path.join(data_folder, "x_test.csv")
x_test =  pd.read_csv(file_path)
file_path = os.path.join(data_folder, "y_test.csv")
y_test =  pd.read_csv(file_path).values.ravel()
logging.info("Read the testing data successfully.")

logging.info("Reading validation dataset........")
file_path = os.path.join(data_folder, "x_valid.csv")
x_valid =  pd.read_csv(file_path)
file_path = os.path.join(data_folder, "y_valid.csv")
y_valid =  pd.read_csv(file_path).values.ravel()
logging.info("Read the validation data successfully.")

#Function to evaluate model performance
def model_eval(Y_test, Y_pred):
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (fp + tn)
    F1_Score = 2 * (recall * precision) / (recall + precision)
    result = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, 'Specificity': specificity, 'F1': F1_Score}
    return result

models = {
    "LogisticRegression": LogisticRegression( solver='liblinear'),
    "K-Neighbors Classifier": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier(),
    "XGBClassifier": XGBClassifier(),
}

model_names = list(models.keys())
model_list = []
accuracy_score_list = []
Recall_score_list = []

for i in range(len(model_names)):
    model = list(models.values())[i]
    model.fit(x_train, y_train.ravel())  # Train model

    # Make predictions
    Y_train_pred = model.predict(x_train)
    Y_test_pred = model.predict(x_test)

    # Evaluate Train and Test dataset
    # Evaluate Train and Test dataset
    model_train_results = model_eval(y_train, Y_train_pred)
    model_test_results = model_eval(y_test, Y_test_pred)

    model_train_accuracy_score = model_train_results["Accuracy"]
    model_train_recall_score = model_train_results["Recall"]

    model_test_accuracy_score = model_test_results["Accuracy"]
    model_test_recall_score = model_test_results["Recall"]


    print(model_names[i])
    model_list.append(model_names[i])
    
    print('Model performance for Training set')
    print("- accuracy_score: {} ".format(model_train_accuracy_score))
    print("- recall_score: {}".format(model_train_recall_score))
    print('----------------------------------')

    print('Model performance for Test set')
    print("- accuracy_score: {}".format(model_test_accuracy_score))
    print("- recall_score: {}".format(model_test_recall_score))

    accuracy_score_list.append(model_test_accuracy_score)

    print('=' * 35)
    print('\n')
# Sort models based on accuracy scores
sorted_models = sorted(zip(model_names, accuracy_score_list), key=lambda x: x[1], reverse=True)

print("Models sorted by accuracy:")
for model in sorted_models:
    print("Model: {}, Accuracy: {}".format(model[0], model[1]))