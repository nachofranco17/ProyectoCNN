import pandas as pd


chestXrays = pd.read_csv('Data/Data_Entry_2017.csv')

chestXrays_clone = chestXrays.copy()

print(chestXrays_clone.columns)




