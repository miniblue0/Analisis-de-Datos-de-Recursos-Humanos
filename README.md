# Análisis de Datos de Recursos Humanos
## Analizar datos de empleados para evaluar la satisfacción laboral y reducir la rotación.
# Tareas:
1. Crear una base de datos que almacene información sobre empleados,
   encuestas de satisfacción y registros de rotación.
2. Realizar un análisis exploratorio de los datos y visualización de patrones con
   pandas y seaborn.
3. Desarrollar modelos de Machine Learning para predecir la probabilidad de
   rotación de empleados.
4. Generar informes de recomendaciones basados en los análisis.
# Configuración de la Base de Datos:
1. Crea una base de datos en SQL Server.
  Diseña las tablas necesarias para almacenar información sobre empleados,
  encuestas de satisfacción y registros de rotación.
2. Extracción y Análisis de Datos con Python:
  Usa Python y bibliotecas como pandas para conectarte a la base de datos y
  extraer datos.
  Realiza análisis exploratorio y visualización de datos.
3. Modelos Predictivos:
  Desarrolla modelos de Machine Learning para predecir la rotación de
  empleados usando scikit-learn.



# Variables en .env:

```env
   DATASET_NAME = "kelissamilano/tech-company-employee-data"
   SERVER = "localhost"
   DATABASE = "HR_Analytics"
   USERNAME = "user"
   PASSWORD = "apassword"
   TABLE_NAME = "Employees"
   CONN_STR = f"mssql+pyodbc://{USERNAME}:{PASSWORD}@{SERVER}/{DATABASE}?driver=ODBC+Driver+17+for+SQL+Server"
   PATH = "datasets/"
   CSV_PATH = "datasets/techstart_employee_data.csv"
```
```sql
   DB_QUERY = f"IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = '{DATABASE}') CREATE DATABASE {DATABASE}"
   CREATE_TABLE_QUERY= f"""
           IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{TABLE_NAME}')
           CREATE TABLE {TABLE_NAME} (
               Employee_ID VARCHAR(50) PRIMARY KEY,
               First_Name NVARCHAR(50),
               Last_Name NVARCHAR(50),
               Gender NVARCHAR(10),
               Age INT,
               Job_Title NVARCHAR(100),
               Department NVARCHAR(100),
               Years_of_Service INT,
               Monthly_Rate DECIMAL(10,2),
               Attrition NVARCHAR(10),
               Overtime NVARCHAR(10),
               Environment_Satisfaction INT,
               Job_Satisfaction INT,
               Work_Life_Balance INT
           );
           """
   DTYPE= {
       'Employee_ID': String(50),
       'First_Name': String(50),
       'Last_Name': String(50),
       'Gender': String(10),
       'Age': Integer(),
       'Job_Title': String(100),
       'Department': String(100),
       'Years_of_Service': Integer(),
       'Monthly_Rate': Float(),
       'Attrition': Boolean(),
       'Overtime': Boolean(),
       'Environment_Satisfaction': Float(),
       'Job_Satisfaction': Float(),
       'Work_Life_Balance': Float()
   }
```
