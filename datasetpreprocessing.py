import pandas as pd
import matplotlib.pyplot as plt

### plotting correlation for the clinical attributes

def plot_corr(dataset):
  f = plt.figure(figsize=(10, 10))
  plt.matshow(dataset.corr(), fignum=f.number)
  plt.show()

### removing attributes with high correlation

def rem_corr(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    #print(col_corr)


def preprocess_dataset(path):

  df = pd.read_csv (path)

  ### converting mutations of genes in boolean
  for col in df.iloc[:, -173:-1].columns:
    for val in col:
      if val != 0 or val != '0':
          val == 1

  ### dropping NaN values and deleting the ID column
  df = df.dropna()
  df = df.iloc[:,1:]

  ### removing attributes with only one value
  for col in df:
    if len(df[col].unique()) == 1:
      del df[col]

  ### converting all "object" types in "category" type, and 
  ### transforming them into numerical 

  df_obj = df.select_dtypes(object)
  for col in df_obj:
    df[col] = df[col].astype('category')
    # printing categories for inspection
    #print(df[col].cat.categories)
    df[col] = df[col].astype('category').cat.codes

  #plot_corr(df)

  #rem_corr(df, 0.5)

  #plot_corr(df)

  ## downsampling majority class

  df0 = df[df.type_of_breast_surgery==0]
  df1 = df[df.type_of_breast_surgery==1]

  df1 = df1.iloc[:len(df0), :]

  df = pd.concat([df0, df1])


  df.info()
  df.to_csv("data/metabric_preprocess.csv")

  ### save dataframe as CSV
  #df.to_csv("data/metabric.csv")

  return df


if __name__ == '__main__':
    preprocess_dataset("data/METABRIC_RNA_Mutation.csv")

