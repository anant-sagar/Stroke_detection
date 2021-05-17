
import streamlit as st
import pickle 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from db import Prediction
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

st.title("Stroke Detection")

#st.image("images\")


st.success('''Our Project  provides an automated ML system to identify patients with acute ischemic stroke within 4.5 hours of symptom onset.
 ML techniques may be feasible and useful in identifying candidates for therapy among patients with unclear stroke onset time.''')


@st.cache()
def load_data(path):
    df= pd.read_csv(path)
    return df

df = load_data("dataset\healthcare-dataset-stroke-data.csv")



def load_model(path ='stroke_detect.pk'):
    with open(path,"rb") as file :
            model = pickle.load(file)
    st.sidebar.info("Model Loaded Sucessfully.")
    return model



def opendb():
    engine = create_engine('sqlite:///db.sqlite3') # connect
    Session =  sessionmaker(bind=engine)
    return Session()

def save_results(name,features,result,output):
    try:
        db = opendb()
        p = Prediction(name=name,features=str(features),result=result,output=output)
        db.add(p)
        db.commit()
        db.close()
        return True
    except Exception as e:
        st.write("database error:",e)
        return False



def  predict(data, model_dict):
    model=model_dict['model']
    encoders=model_dict['encoders']
    dummy_gender = encoders[0][1].transform(data[['gender']]).toarray()
    dummy_ever_married = encoders[1][1].transform(data[['ever_married']]).toarray()
    dummy_work_type = encoders[2][1].transform(data[['work_type']]).toarray()
    dummy_Residence_type = encoders[3][1].transform(data[['Residence_type']]).toarray()
    dummy_smoking_status = encoders[4][1].transform(data[['smoking_status']]).toarray()
    data = pd.concat([data.drop(['gender'],axis = 1),pd.DataFrame(dummy_gender)],axis =1)
    data = pd.concat([data.drop(['ever_married'],axis = 1),pd.DataFrame(dummy_ever_married)],axis =1)
    data = pd.concat([data.drop(['work_type'],axis = 1),pd.DataFrame(dummy_work_type)],axis =1)
    data = pd.concat([data.drop(['Residence_type'],axis = 1),pd.DataFrame(dummy_Residence_type)],axis =1)
    data = pd.concat([data.drop(['smoking_status'],axis = 1),pd.DataFrame(dummy_smoking_status)],axis =1)
    result = model.predict(data.values)
    return result


    
if st.checkbox("About"):
    st.markdown("""According to the WHO, cerebrovascular stroke is the second largest cause of death
worldwide, accounting for more than 5 million deaths in 2017 alone. Conventional detection
methods involve physician administered exams including blood tests, angiograms, carotid
ultrasounds, CT, and MRI scans. These neurological scans such as MRI are first of all costly, and
are generally obtained 12 hours after the onset of symptoms, thus preventing patients from
receiving treatment in the optimal timeframe. Thus, early, automated diagnosis methods using
other biomarkers of stroke have been a central focus on cerebrovascular accident (CVA)
research. Currently, there are no existing portable stroke screening platform available to the
public. This is a dire issue requiring immediate action, since over 140,000 people die of stroke in
the US and in every 40 seconds, someone is afflicted with stroke. Since current diagnosis
methods yield accuracies up to 80% and stroke diagnosis is largely inaccessible in areas without
access to quality medical facilities, a holistic patient centered early diagnosis application is
critical to address this pressing issue. Moreover, recent advances in the applications of deep
learning and computer vision, backend databases, and mobile application development allow us
to engineer a timely, inexpensive, accurate stroke detection platform with far higher accuracy
than current medical professionals.""")

if  st.checkbox("Make Prediction"):
    model_dict = load_model()

    name = st.text_input("Enter Name")



    col1, col2 = st.beta_columns(2)

    with col1:
        gender = st.radio("gender",["Male","Female","Other"])

    with col2:
        age = st.number_input("Age",min_value=10)




    col3, col4 = st.beta_columns(2)

    with col3:
        hypertension = st.radio("Hypertension",["Yes","No"])

    with col4:
        heart_disease = st.radio("Heart paitent",["Yes","No"]) 



    
    col5, col6 = st.beta_columns(2)

    with col5:
        married = st.radio("Married",["Yes","No"])

    with col6:
        work_type = st.selectbox("Occupation",['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])



    col7, col8 = st.beta_columns(2)

    with col7:
        bmi = st.number_input("BMI")

    with col8:
        Residence_type = st.selectbox("Residence Type",['Urban', 'Rural'])


    col9, col10 = st.beta_columns(2)

    with col9:
        avg_glucose_level = st.number_input("Avg Glucose Level")

    with col10:
        smoking_status = st.selectbox("Smoking Status",['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

    if st.sidebar.button('Predict')and name:
        heart_disease=1 if heart_disease == "Yes" else 0
        hypertension=1 if hypertension == "Yes" else 0
        


    
        
        features = np.array([[gender,
 age,
 hypertension,
 heart_disease,
 married,
 work_type,
 Residence_type,
 avg_glucose_level,
 bmi,
 smoking_status]])
        
        
        cols=['gender',
 'age',
 'hypertension',
 'heart_disease',
 'ever_married',
 'work_type',
 'Residence_type',
 'avg_glucose_level',
 'bmi',
 'smoking_status']
        df = pd.DataFrame(features,columns=cols)
        result = predict(df, model_dict)
        if result[0] == 0:
            st.sidebar.info("your data predict you are healthy")
            s = save_results(name,features,result,"healthy")

        else:
            st.sidebar.info("You might have chaces of a stroke")
            s = save_results(name,features,result,"chances of stroke")



if st.checkbox("View Records"):
    db = opendb()
    results = db.query(Prediction).all()
    db.close()
    record = st.selectbox("select a cutomer Record",results)
    if record:
        st.write(record.name)
        st.write(record.output)
        st.write(record.created_on)
        st.write(record.features)


if st.checkbox("Visualization"):
    visualization= st.selectbox("Training Data Graphs",['All Data Visualise',
 'Gender',
 'Age',
 'Hypertension',
 'Heart Disease',
 'Ever Married',
 'Work Type',
 'Residence Type',
 'Average Glucose Level',
 'BMI',
 'Smoking Status'])
    if  visualization=="All Data Visualise":
        st.image("images/visualise.png")

    if  visualization=="Gender":
        st.image("images/gender.png")

    if  visualization=="Heart Disease":
        st.image("images/heart.png")

    if  visualization=="Ever Married":
        st.image("images/married.png")

    if  visualization=="Hypertension":
        st.image("images/hyper.png")

    if  visualization=="Resisdence Type":
        st.image("images/resisdance.png")
        
    if  visualization=="Work Type":
        st.image("images/work.png")

    if  visualization=="Smoking Status":
        st.image("images/smoke.png")

    if  visualization=="BMI":
        st.image("images/bmi.png")

    if  visualization=="Age":
        st.image("images/age.png")

    if  visualization=="Average Glucose Level":
        st.image("images/glucose.png")

    