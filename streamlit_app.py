import streamlit as st

st.title('🎈 App Name')

import pandas as pd

# Create a sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)

# Display the DataFrame
st.dataframe(df)

# Display a static table
st.table(df)

st.write('Hello world!')
