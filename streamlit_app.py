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


cat_cols = df.select_dtypes(include=['object']).columns.tolist()

num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()


st.write(cat_cols)
st.write(num_cols)

slider_val = {}

st.write("### Numerical Features")
for i in num_cols:
    min_val = float(df[i].min())
    max_val = float(df[i].max())

    slider_val[i] = st.slider(
                    f"{i}",  # Label for the slider
                    min_val,  # Minimum value
                    max_val,  # Maximum value
                    # (min_val + max_val) / 2  # Default value (midpoint)
                    )

st.write("### Categorical Features")

cat_input = {}

for i in cat_cols:
    unique = df[i].unique().tolist()
    cat_input[i] = st.radio(
                        f"**{i}**",
                        unique
                    )
    
st.write("###### *asumsi tidak ada hierarki pada kelas")

if __name__ == "__main__":
    main()