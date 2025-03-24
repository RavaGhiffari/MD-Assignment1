import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
import matplotlib.pyplot as plt
import plotly.express as px

def main():

    st.title('🎈 Classification App')

    saved_data = jb.load('trained_model_used.pkl')
    model = saved_data['model']
    ohe = saved_data['ohe']
    label_mapping = saved_data['label_mapping']

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

    input_data = pd.DataFrame([user_inputs])
    
    st.write("OneHotEncoder expects:", ohe.feature_names_in_)
    st.write("Input Data Columns Before Encoding:", input_data.columns)



   # Add a Classify button
    if st.button("Classify"):
        # Convert user inputs into a DataFrame
        st.table(input_data)
        input_data['Gender'] = input_data['Gender'].map({"Male": 0, "Female": 1})
        input_data['family_history_with_overweight'] = input_data['family_history_with_overweight'].map({"Yes": 1, "No": 0})
        input_data['FAVC'] = input_data['FAVC'].map({"Yes": 1, "No": 0})
        input_data['SMOKE'] = input_data['SMOKE'].map({"Yes": 1, "No": 0})
        input_data['SCC'] = input_data['SCC'].map({"Yes": 1, "No": 0})

        encoded_caec = ohe.transform(input_data[['CAEC']])
        encoded_calc = ohe.transform(input_data[['CALC']])
        encoded_mtrans = ohe.transform(input_data[['MTRANS']])

        encoded_caec_df = pd.DataFrame(encoded_caec, columns=ohe.get_feature_names_out(['CAEC']))
        encoded_calc_df = pd.DataFrame(encoded_calc, columns=ohe.get_feature_names_out(['CALC']))
        encoded_mtrans_df = pd.DataFrame(encoded_mtrans, columns=ohe.get_feature_names_out(['MTRANS']))
        
        input_data = input_data.drop(columns=['CAEC', 'CALC','MTRANS'])

        input_data = pd.concat([input_data, encoded_caec_df, encoded_calc_df, encoded_mtrans_df], axis=1)


        


        column_order = [
        'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 
        'FAVC', 'FCVC', 'NCP', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 
        'CAEC_No', 'CAEC_Sometimes', 'CAEC_Frequently', 'CAEC_Always',
        'CALC_No', 'CALC_Sometimes', 'CALC_Frequently','MTRANS'
        ]

        input_data = input_data[column_order]

        prediction = model.predict(input_data)

        st.write("### Prediction Result")
        st.write(f"**Predicted Obesity Class:** {list(label_mapping.keys())[list(label_mapping.values()).index(prediction[0])]}")

if __name__ == "__main__":
    main()