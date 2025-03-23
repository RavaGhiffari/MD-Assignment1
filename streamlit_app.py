import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
import matplotlib.pyplot as plt

def main():

    st.title('🎈 Classification App')

    model = jb.load('trained_model.pkl')

    df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
    
    with st.expander("Data Overview"):
    # Display the head of the DataFrame inside the expander
        st.write('#### This is the raw data')
        st.dataframe(df)

    with st.expander("### Scatter Plot: Height vs Weight"):

        fig, ax = plt.subplots()
        ax.scatter(df['Height'], df['Weight'], alpha=0.5)  # Scatter plot
        ax.set_xlabel('Height (cm)')  # X-axis label
        ax.set_ylabel('Weight (kg)')  # Y-axis label
        ax.set_title('Height vs Weight')  # Title of the plot
        st.plt(fig)  # Display the plot in Streamlit

    column_name = st.selectbox(
        "Select a column to display unique values",  # Label for the dropdown
        df.select_dtypes(include=['object']).columns  # List of options (columns in the DataFrame)
    )

    # Display unique values from the selected column
    if column_name:
        st.write(f"### Unique values in column '{column_name}':")
        st.write(df[column_name].unique())


    cat_cols = df.select_dtypes(include=['object']).drop(columns=['NObeyesdad']).columns.tolist()

    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()


    st.write(cat_cols)
    st.write(num_cols)

    slider_val = {}

    st.write("### Numerical Features")
    for i in num_cols:
        if i in ['Age', 'FCVC','NCP', 'CH2O', 'FAF', 'TUE']:
            min_val = int(df[i].min())
            max_val = int(df[i].max())

            slider_val[i] = st.slider(
                f"{i}",min_val, max_val, step=1
            )

        else:
            min_val = float(df[i].min())
            max_val = float(df[i].max())

            slider_val[i] = st.slider(
                            f"{i}",  # Label for the slider
                            min_val,  # Minimum value
                            max_val,  # Maximum value
                        # (min_val + max_val) / 2,
                            )

    st.write("### Categorical Features")

    cat_input = {}

    for i in cat_cols:
        unique = df[i].unique().tolist()
        if i == 'family_history_with_overweight':
            cat_input[i] = st.radio(
                f"**Genetic Weight Factors?**",
                unique
            )
        else:
            cat_input[i] = st.radio(
                                f"**{i}**?",
                                unique
                            )
    
    st.write("###### *Assuming No hierarchical order in class")

    user_input = {**slider_val, **cat_input}

    st.button('Classify')

if __name__ == "__main__":
    main()