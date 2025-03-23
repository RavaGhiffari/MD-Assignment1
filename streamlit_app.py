import streamlit as st

st.title('🎈 App Name')

import pandas as pd

# Create a sample DataFrame

df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# Display the DataFrame
st.dataframe(df)

st.markdown(df)

# Display a static table
st.table(df)

st.write('Hello world!')
