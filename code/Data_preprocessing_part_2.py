# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler 
import logging
import os

# Create a folder for log files if it doesn't exist
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)
# Specify the path of the data folder
data_folder = os.path.join(os.getcwd(), "code", "data")

# Configure logging for Data_preprocessing_part_2.log
log_file_part_2 = os.path.join(log_folder, "Data_preprocessing_part_2.log")
logging.basicConfig(filename=log_file_part_2, level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")

# Reading the dataset
logging.info("Reading the dataset...")
file_path = r"C:\project\thyroid_detection_main\code\data\data_processed_1.csv"
df = pd.read_csv(file_path)
logging.info("Read the dataset successfully.")

# Replacing "?" with numpy null values
df.replace({"?": np.nan}, inplace=True)
logging.info("Replaced all '?' with null values.")

# Dropping columns with high missing values
df.drop(columns=["TBG", "T3"], inplace=True)
logging.info("Dropped columns 'TBG' and 'T3' due to high missing values.")

# Filling null values in 'sex' with "unknown"
df.sex.fillna("unknown", inplace=True)

# Converting datatype of continuous features to numeric type
numeric_columns = ["TSH", "TT4", "T4U", "FTI"]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
logging.info("Converted continuous features to numeric type.")

# Removing outliers in 'age' feature
index_age = df[df["age"]>100].index
df.drop(index_age, inplace=True)
logging.info("Removed outliers from 'age' feature.")

# removing TSH value higher than 15. That's quiet rare.
index_tsh = df[df["TSH"]>15].index
df.drop(index_tsh, inplace=True)
logging.info("Droped outliers from TSH feature.")

numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

print('We have {} numerical features: {}'.format(len(numeric_features), numeric_features))
print('We have {} categorical features: {}'.format(len(categorical_features), categorical_features))


# Encoding categorical features
df_encoded = pd.get_dummies(df)
logging.info("Encoded categorical features.")

# Imputing missing values using KNNImputer
def Imputation(df):
    imputer = KNNImputer(n_neighbors=3)
    df_1 = imputer.fit_transform(df)
    df_2 = pd.DataFrame(df_1, columns=df.columns)
    return df_2
df_final = Imputation(df_encoded[:7000])   
logging.info("Imputed missing values using KNNImputer in train and test.")


# Splitting the data into train, test and validation to prevent data leakage.
x_train, x_test, y_train, y_test = train_test_split(df_final.drop(columns="outcome"), df_final["outcome"], test_size=0.2)
validation_data = df_encoded[7000:]

valid_final = Imputation(validation_data)
logging.info("Created training, testing and validation dataset.")


# Fixing imbalanced data by oversampling
def balance_data(x, y):    
    ros = RandomOverSampler(random_state=42)
    x_sample, y_sample = ros.fit_resample(x, y)
    return x_sample, y_sample

x_train, y_train = balance_data(x_train, y_train)
x_test, y_test = balance_data(x_test, y_test)
x_valid, y_valid = balance_data(valid_final.drop(columns="outcome"), valid_final["outcome"])

# Saving the data
logging.info("saving the training data.....")
x_train.to_csv(os.path.join(data_folder, "x_train.csv"), index=False)
y_train.to_csv(os.path.join(data_folder, "y_train.csv"), index=False)
logging.info("Successfully saved the training data.")

logging.info("saving the testing data.....")
x_test.to_csv(os.path.join(data_folder, "x_test.csv"), index=False)
y_test.to_csv(os.path.join(data_folder, "y_test.csv"), index=False)
logging.info("Successfully saved the testing data.")

logging.info("Saving the validation data...")
x_valid.to_csv(os.path.join(data_folder, "x_valid.csv"), index=False)
y_valid.to_csv(os.path.join(data_folder, "y_valid.csv"), index=False)
logging.info("Successfully saved the validation data.")