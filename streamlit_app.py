import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
import matplotlib.pyplot as plt
import plotly.express as px

def main():

    st.title('🎈 Classification App')

    model = jb.load('trained_model.pkl')

    df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
    
    with st.expander("Data Overview"):
    # Display the head of the DataFrame inside the expander
        st.write('#### This is the raw data')
        st.dataframe(df)

    with st.expander("### Scatter Plot: Height vs Weight"):
        fig = px.scatter(
        df, 
        x='Height', 
        y='Weight', 
        color='NObeyesdad',  # Color by NObeyesdad class
        title='Height vs Weight (Colored by Obesity Class)', 
        labels={'Height': 'Height (m)', 'Weight': 'Weight (kg)'}
    )
        fig.update_xaxes(range=[0, df['Height'].max() + 0.1])  # X-axis starts from 0
        fig.update_yaxes(range=[0, df['Weight'].max() + 10])  # Y-axis starts from 0

        st.plotly_chart(fig)  # Display the plot in Streamlit



    # column_name = st.selectbox(
    #     "Select a column to display unique values",  # Label for the dropdown
    #     df.select_dtypes(include=['object']).columns  # List of options (columns in the DataFrame)
    # )

    # Display unique values from the selected column
    # if column_name:
    #     st.write(f"### Unique values in column '{column_name}':")
    #     st.write(df[column_name].unique())


    # cat_cols = df.select_dtypes(include=['object']).drop(columns=['NObeyesdad']).columns.tolist()

    # num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()


    # # st.write(cat_cols)
    # # st.write(num_cols)

    # slider_val = {}

    # st.write("### Numerical Features")
    # for i in num_cols:
    #     if i in ['Age', 'FCVC','NCP', 'CH2O', 'FAF', 'TUE']:
    #         min_val = int(df[i].min())
    #         max_val = int(df[i].max())

    #         slider_val[i] = st.slider(
    #             f"{i}",min_val, max_val, step=1
    #         )

    #     else:
    #         min_val = float(df[i].min())
    #         max_val = float(df[i].max())

    #         slider_val[i] = st.slider(
    #                         f"{i}",  # Label for the slider
    #                         min_val,  # Minimum value
    #                         max_val,  # Maximum value
    #                     # (min_val + max_val) / 2,
    #                         )

    # st.write("### Categorical Features")

    # cat_input = {}

    # for i in cat_cols:
    #     unique = df[i].unique().tolist()
    #     if i == 'family_history_with_overweight':
    #         cat_input[i] = st.radio(
    #             f"**Genetic Weight Factors?**",
    #             unique
    #         )
    #     else:
    #         cat_input[i] = st.radio(
    #                             f"**{i}**?",
    #                             unique
    #                         )
    
    st.write("### Enter Your Details")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 0,  100, 25)
    height = st.slider("Height (m)", 1.0, 2.5, 1.7)
    weight = st.slider("Weight (kg)", 30, 150, 70)
    family_history = st.radio("Family History with Overweight", ["Yes", "No"])
    favc = st.radio("Do you eat high-caloric food frequently?", ["Yes", "No"])
    fcvc = st.slider("Frequency of vegetable consumption (1-3)", 1, 3, 2)
    ncp = st.slider("Number of main meals (1-4)", 1, 4, 3)
    caec = st.radio("Consumption of food between meals", ["No", "Sometimes", "Frequently", "Always"])
    caec = st.selectbox("Consumption of food between meals", ["No", "Sometimes", "Frequently", "Always"])
    smoke = st.radio("Do you smoke?", ["Yes", "No"])
    ch2o = st.slider("Daily water intake (1-3)", 1, 3, 2)
    scc = st.radio("Do you monitor calories?", ["Yes", "No"])
    faf = st.slider("Physical activity frequency (0-3)", 0, 3, 1)
    tue = st.slider("Time using electronic devices (0-2)", 0, 2, 1)
    calc = st.selectbox("Alcohol consumption", ["No", "Sometimes", "Frequently"])
    mtrans = st.selectbox("Transportation method", ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])

    # st.write("###### *Assuming No hierarchical order in class")
    user_inputs = {
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history_with_overweight": family_history,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans
    }

    # st.button('Classify')

   # Add a Classify button
    if st.button("Classify"):
        # Convert user inputs into a DataFrame
        input_data = pd.DataFrame([user_inputs])
        st.table(input_data)

        # Preprocess categorical variables using the saved encoders
        # input_data['Gender'] = input_data['Gender'].map({"Male": 0, "Female": 1})
        # input_data['family_history_with_overweight'] = input_data['family_history_with_overweight'].map({"Yes": 1, "No": 0})
        # input_data['FAVC'] = input_data['FAVC'].map({"Yes": 1, "No": 0})
        # input_data['SMOKE'] = input_data['SMOKE'].map({"Yes": 1, "No": 0})
        # input_data['SCC'] = input_data['SCC'].map({"Yes": 1, "No": 0})

        # # Apply OneHotEncoder to CAEC and CALC
        # encoded_caec = ohe.transform(input_data[['CAEC']])
        # encoded_calc = ohe.transform(input_data[['CALC']])

        # # Convert encoded arrays to DataFrames
        # encoded_caec_df = pd.DataFrame(encoded_caec, columns=ohe.get_feature_names_out(['CAEC']))
        # encoded_calc_df = pd.DataFrame(encoded_calc, columns=ohe.get_feature_names_out(['CALC']))

        # # Drop original columns and concatenate encoded DataFrames
        # input_data = input_data.drop(columns=['CAEC', 'CALC'])
        # input_data = pd.concat([input_data, encoded_caec_df, encoded_calc_df], axis=1)

        # # Ensure the column order matches the training data
        # column_order = [
        #     'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 
        #     'FAVC', 'FCVC', 'NCP', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 
        #     'CAEC_No', 'CAEC_Sometimes', 'CAEC_Frequently', 'CAEC_Always',
        #     'CALC_No', 'CALC_Sometimes', 'CALC_Frequently'
        # ]
        # input_data = input_data[column_order]

        # # Make predictions
        # prediction = model.predict(input_data)  # Predict the class
        # prediction_proba = model.predict_proba(input_data)  # Predict probabilities (if applicable)

        # # Display the prediction
        # st.write("### Prediction Result")
        # st.write(f"**Predicted Obesity Class:** {prediction[0]}")

        # # Display prediction probabilities (if applicable)
        # if hasattr(model, "predict_proba"):
        #     st.write("### Prediction Probabilities")
        #     proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
        #     st.write(proba_df)


if __name__ == "__main__":
    main()