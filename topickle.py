import pandas as pd

csvPath = './tenYears.xlsx'

df = pd.read_excel(csvPath)

df.to_pickle('./pickle/tenYears.pkl')