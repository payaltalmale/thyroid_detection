import pandas as pd
import logging
import os

# Create a folder for log files 
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)
# Specify the path of the data folder
data_folder = os.path.join(os.getcwd(), "code", "data")


# Configure logging
log_file = os.path.join(log_folder, "Data_preprocessing_part_1.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

logging.info("Reading the dataset.........")
file_path = os.path.join(data_folder, "thyroid.csv")
df = pd.read_csv(file_path)
logging.info("Read the dataset successfully.")

# Saving the first character in a new column. Because that's what matters for this problem statement.
logging.info("Data preprocessing part 1 started.........")
df["outcome"] = df["class"].str[0]
df.drop(columns="class", inplace=True)
logging.info("Extracted valuable information from the target variable.")

# Replacing all possible disease outcomes into one category - "yes".
disease_outcomes = ['S', 'F', 'A', 'R', 'I', 'M', 'N', 'G', 'K', 'L', 'Q', 'J', 'C', 'O', 'H', 'D', 'P', 'B', 'E']
df['outcome'].replace(to_replace=disease_outcomes, value="yes", inplace=True)
logging.info("Replaced all thyroidal diseases into one category.")

# Replacing the binary outputs into integer values 0 and 1 for simplicity.
df['outcome'].replace({"-": 0, "yes": 1}, inplace=True)
logging.info("Classified the target features into 0 and 1.")

logging.info("Saving the processed data........")
processed_data_file = os.path.join(data_folder, "data_processed_1.csv")
df.to_csv(processed_data_file, index=False)
logging.info("Successfully saved the data in a CSV file.")

logging.info("Data Preprocessing part 1 complete.")
