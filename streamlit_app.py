import streamlit as st

st.title('🎈 App Name')

import pandas as pd


df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
with st.expander("Data Overview"):
    # Display the head of the DataFrame inside the expander
    st.write('#### This is the raw data')
    st.dataframe(df)

column_name = st.selectbox(
    "Select a column to display unique values",  # Label for the dropdown
    df.select_dtypes(include=['object']).columns  # List of options (columns in the DataFrame)
)

# Display unique values from the selected column
if column_name:
    st.write(f"### Unique values in column '{column_name}':")
    st.write(df[column_name].unique())


