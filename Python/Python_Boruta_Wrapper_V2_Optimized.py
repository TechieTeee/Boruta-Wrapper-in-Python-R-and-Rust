import pandas as pd
!pip install boruta
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from google.colab import files

def load_data():
  #Load CSV File of Data into the Google Colab Environment
  upload = files.upload()

  #Load the dataset
  df = pd.read_csv('sales.csv')
  
  return df

def separate_target_variable(func):
  def wrapper(*args, **kwargs):
    df = func(*args, **kwargs)
    target = df['target']
    df = df.drop(columns=['target'])
    return df, target
  return wrapper

@separate_target_variable
def preprocess_data():
  df = load_data()
  return df

def create_boruta_selector(rf):
  #Create the Boruta feature selector object
  boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
  return boruta_selector

def fit_boruta_selector(df, target, rf):
  #Fit the Boruta selector to the dataset
  boruta_selector = create_boruta_selector(rf)
  boruta_selector.fit(df.values, target.values)
  return boruta_selector

def print_selected_features(df, boruta_selector):
  #Print the selected features
  selected_features = df.columns[boruta_selector.support_].to_list()
  print('Selected features:', selected_features)

def main():
  #Define Random Forest parameters
  rf_params = {'n_jobs': -1, 'class_weight': 'balanced', 'max_depth': 5}

  #Create a Random Forest classifier object
  rf = RandomForestClassifier(**rf_params)

  #Preprocess the data
  df, target = preprocess_data()

  #Create and fit Boruta selector
  boruta_selector = fit_boruta_selector(df, target, rf)

  #Print selected features
  print_selected_features(df, boruta_selector)

if __name__ == '__main__':
  main()
