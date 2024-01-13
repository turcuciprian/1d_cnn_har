from pandas import read_csv

# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace==True)
    return dataframe.values
