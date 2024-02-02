from datetime import datetime
import pandas as pd
import os

def export_dataframe_to_csv(dataframe: pd.DataFrame, path: str, file_name: str):
    ''' Exports the given dataframe to a csv file '''
    current_date = datetime.now().date()
    current_hour = datetime.now().time().hour
    current_minute = datetime.now().time().minute

    # Check if folder exists and create it if not
    if not os.path.exists(path):
        os.makedirs(path)

    # If path does not end with a slash, add one
    if not path.endswith("/"):
        path = f"{path}/"

    dataframe.to_csv(f"{path}{file_name}_{current_date}_{current_hour}-{current_minute}.csv", index=False)

def export_predictions_to_file(predictions_df: pd.DataFrame, path: str, filename: str, prediction_column: str):
    ''' Exports the predictions from the given dataframe to a file'''
    # Excape newlines in prediction
    predictions = [x.replace("\n", " ") for x in predictions_df[prediction_column]]

    # Check if folder exists and create it if not
    if not os.path.exists(path):
        os.makedirs(path)

    # If path does not end with a slash, add one
    if not path.endswith("/"):
        path = f"{path}/"

    with open(f"{path}{filename}", "w") as f:
        for prediction in predictions:
            f.write(f"{prediction}\n")