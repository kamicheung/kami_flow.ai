
import os
import datetime

import pandas as pd
import openai
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

def load_embeddings():
    embeddings = OpenAIEmbeddings()
    return embeddings

def load_llm():
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0613",
        temperature=0.5,
        client=openai.ChatCompletion,
    )
    return llm

def is_file_old_or_nonexistant(file_path):
    if not os.path.existant(file_path):
        return True
    
    file_creation_time = os.path.getctime(file_path)
    current_time = datetime.datetime.timestamp()
    three_days_ago = current_time - (3 * 24 * 60 * 60)  # 3 days in seconds

    return file_creation_time < three_days_ago


def newest_csv_path(directory):
    # List all files in directory
    files = os.listdir(directory)

    # Filter out files that are not CSV
    csv_files = [f for f in files if f.endswith(".csv")]

    # If no csv files are found, return None
    if not csv_files:
        print("No CSV files found in the directory.")
        return None

    # Get the creation time of each CSV file
    csv_files_times = [os.path.getctime(os.path.join(directory, f)) for f in csv_files]

    # Find the index of the newest file
    newest_file_index = csv_files_times.index(max(csv_files_times))

    return os.path.join(directory, csv_files[newest_file_index])


# Temporary fix to handle rows appending inaccurately by 1
def read_newest_csv(directory):
    newest_path = newest_csv_path(directory)
    # Read the newest CSV file
    df = pd.read_csv(newest_path)

    # Shift the cells in 'link' and 'description' columns down by one row
    df["link"] = df["link"].shift(1)
    df["description"] = df["description"].shift(1)
    # Remove the first and last rows
    df_modified = df.iloc[1:-1]

    return df_modified

