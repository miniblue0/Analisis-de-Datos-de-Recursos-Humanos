
###1. Configuracion de la Base de Datos:

import os
from kaggle.api.kaggle_api_extended import KaggleApi
from sqlalchemy import create_engine,text#, Integer, String, Float, Boolean
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
PATH =os.getenv("PATH")
CSV_PATH =os.getenv("CSV_PATH")
DATASET_NAME = os.getenv("DATASET_NAME")
SERVER = os.getenv("SERVER")
TABLE_NAME = os.getenv("TABLE_NAME")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
CONN_STR = os.getenv("CONN_STR")
DB_QUERY = os.getenv("DB_QUERY")
CREATE_TABLE_QUERY=os.getenv("CREATE_TABLE_QUERY")
DTYPE=os.getenv("DTYPE")

def extract_dataset():
    os.makedirs(PATH, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET_NAME,path=PATH, unzip=True)

def create_db_or_table():
    engine = create_engine(f"mssql+pyodbc://{USERNAME}:{PASSWORD}@{SERVER}/master?driver=ODBC+Driver+17+for+SQL+Server", isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        conn.execute(text(DB_QUERY))
        print("conectado a la base de datos")

    engine = create_engine(CONN_STR)
    with engine.connect() as conn:
        conn.execute(text(CREATE_TABLE_QUERY))
    print("tabla creada")

def load_to_sql():
    df = pd.read_csv(CSV_PATH)
    df = df[[
        "Employee_ID", "First_Name", "Last_Name", "Gender", "Age", "Job_Title",
        "Department", "Years_of_Service", "Monthly_Rate", "Attrition",
        "Overtime", "Environment_Satisfaction", "Job_Satisfaction", "Work_Life_Balance"
    ]]
    df['Attrition'] = df['Attrition'].replace({'Yes': True, 'No': False})
    df['Overtime'] = df['Overtime'].replace({'Yes': True, 'No': False})

    df = df.where(pd.notna(df), None)
    engine = create_engine(CONN_STR)

    with engine.connect() as conn:
        df.to_sql(TABLE_NAME, con=conn, if_exists="append", index=False, dtype= DTYPE)
    print("cargado")



##2.Extraccion y Analisis de Datos con Python:


import matplotlib.pyplot as plt

def extract_data_from_sql():
    engine = create_engine(CONN_STR)
    query = f"SELECT * FROM {TABLE_NAME}"
    df = pd.read_sql(query, con=engine)
    return df


def plot_attrition_distribution(df):

    attrition_counts = df['Attrition'].value_counts()
    attrition_labels = ['Se quedo', 'Se fue']  #guardo los valores para el grafico y nombres para comparar

    plt.figure(figsize=(6, 6))
    plt.pie(attrition_counts, labels=attrition_labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff6666'])  #creo el grafico de torta con los valores guardados
    plt.title('Distribución de la rotación de empleados')
    plt.axis('equal')
    plt.show()




def perform_eda(df):
    #print(df.describe())
    #print(df.info())
    plot_attrition_distribution(df)


def process_data(df):
    numeric_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()  #tomo las columnas numericas
    df[numeric_cols]= df[numeric_cols].apply(lambda col: col.fillna(col.median()))  #reemplazo valores faltantes,nulos,etc. con la media

    df = pd.get_dummies(df, columns=['Gender','Department','Job_Title'],drop_first=True) ## reformateo las columnas a valores binarios

    df['Attrition'] = df['Attrition'].astype(int) #marco attrition como entero
    return df



## 3. Predecir la rotacion de empleados con sklearn

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def oversampling_predict_model(df):


    x = df.drop(columns=['Employee_ID','First_Name','Last_Name','Attrition']) #descarta las columnas y toma las demas como predictorasw
    y = df['Attrition'] #usa attrition como objetivo, true o false si se queda o s eva

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)     # usa 80% para entrenar y 20% para evaluar



    smote = SMOTE(sampling_strategy=0.5, random_state=42)  #agregue el smote para tratar de balancear los datos en el entrenamiento (el menor va a tener el 50% del mayor)
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train,y_train)


    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)   #use randomForest para entrenarlo

    model.fit(x_train_balanced, y_train_balanced)

    y_pred = model.predict(x_test) #el modelo hace predicciones

    print('\n reporte de clasificacion:')
    print(classification_report(y_test, y_pred))
    '''                      presision   recall             
     false(se quedaron)      0.85 (85%)   0.90(92% correctamente clasif)
     true(se fueron)         0.75 (75%)   0.65(65% corectamente clasif)
     
     accuracy                                                              0.84(84% son correctas)
     '''



    print('\n matriz de confusion')
    print(confusion_matrix(y_test,y_pred))

    #muestra cuantas predicciones fueron correctas
    ''' se quedaron:                               se fueron:
        [[720 pred. positivos(se quedaron)         80 pred. falso positivo()]  
        [70  pred. falso neg. (no se quedaron)     130  pred. positiva (se fueron)]]'''





def main():
    #1 extraer datos de la api y cargarlos a sql
    extract_dataset()
    create_db_or_table()
    load_to_sql()
    print("datos cargados a sql")
    #2 extraer datos de sql y prepararlos para analizar
    df = extract_data_from_sql()
    perform_eda(df)
    #3 hacer predicciones con sklearn
    df = process_data(df)
    oversampling_predict_model(df)
    #print(df)

if __name__ == "__main__":
    main()
