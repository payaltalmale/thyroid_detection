# Importing the necessary libraries
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
import logging
import os

# Create a folder for log files if it doesn't exist
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)
# Specify the path of the data folder
data_folder = os.path.join(os.getcwd(), "code", "data")


# Configure logging for Data_preprocessing_part_2.log
log_file_part_3 = os.path.join(log_folder, "Feature_selection.log")
logging.basicConfig(filename=log_file_part_3, level=logging.INFO, format='%(asctime)s %(message)s',
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


# Applying selectKbest() to reduce the number of features.

def feature_selection(x,y):
   
    obj = SelectKBest(chi2, k=4)
    obj.fit_transform(x,y)
    filter = obj.get_support()
    feature = x.columns
    final_f = feature[filter]
    logging.info(final_f)
    print(final_f)
    
    return final_f

logging.info("4 best features are............")
features = feature_selection(x_train, y_train)
logging.info(features)