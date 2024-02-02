import json
import random
import pandas as pd

def get_text_from_file(file_path: str) -> str:
  ''' Returns the content of the given file '''
  with open(file_path,encoding='utf-8') as f:
    text = f.read()
    return text

def get_category_from_entry(data_set, entry_number): 
    return data_set[entry_number][f"{entry_number + 1}"]["category"]

def get_modifiedtripleset_from_entry(data_set, entry_number): 
    return data_set[entry_number][f"{entry_number + 1}"]["modifiedtripleset"]

def get_originaltripleset_from_entry(data_set, entry_number): 
    return data_set[entry_number][f"{entry_number + 1}"]["originaltriplesets"]["originaltripleset"]

def get_lexicalisations_from_entry(data_set, entry_number): 
    return data_set[entry_number][f"{entry_number + 1}"]["lexicalisations"]

def get_triplesetsize_from_entry(data_set, entry_number):
    return data_set[entry_number][f"{entry_number + 1}"]["size"]

def add_webnlg_train_entry_to_df(data_set, entry_number, dataframe):
    category = get_category_from_entry(data_set, entry_number)
    modifiedtripleset = get_modifiedtripleset_from_entry(data_set, entry_number)
    originaltripleset = get_originaltripleset_from_entry(data_set, entry_number)
    lexicalisations = get_lexicalisations_from_entry(data_set, entry_number)
    triplesetsize = get_triplesetsize_from_entry(data_set, entry_number)
    new_row = pd.Series([
                            category,
                            modifiedtripleset,
                            originaltripleset,
                            lexicalisations,
                            triplesetsize,
                            entry_number + 1
        ], index=dataframe.columns)
    return pd.concat([dataframe, pd.DataFrame([new_row])], ignore_index=True)

def create_webnlg_df(file_path: str, sample_size: int, start_id: int = 1) -> pd.DataFrame:
    webnlg_df = pd.DataFrame(columns=['category', 'modifiedtripleset', 'originaltripleset', 'lexicalisations', 'triplesetsize', 'id'])
    data_set = json.loads(get_text_from_file(file_path))["entries"]
    entry_number = start_id - 1
    
    for i in range (sample_size):
        webnlg_df = add_webnlg_train_entry_to_df(data_set, entry_number + i, webnlg_df)
    return webnlg_df

def create_webnlg_random_df(file_path: str, dataset_size: int, sample_size: int) -> pd.DataFrame:
    webnlg_df = pd.DataFrame(columns=['category', 'modifiedtripleset', 'originaltripleset', 'lexicalisations', 'triplesetsize', 'id'])
    data_set = json.loads(get_text_from_file(file_path))["entries"]
    
    # Generate a list of random numbers between 0 and train_set_size with seed 42
    random.seed(42)
    random_numbers = random.sample(range(0, dataset_size), sample_size)

    for i in random_numbers:
        webnlg_df = add_webnlg_train_entry_to_df(data_set, i, webnlg_df)
    return webnlg_df

def create_dataframe(file_path: str, dataset_size: int, sample_size: int, random_sample: bool):
    if random_sample:
        return create_webnlg_random_df(file_path, dataset_size, sample_size)
    else:
       return create_webnlg_df(file_path, sample_size)
