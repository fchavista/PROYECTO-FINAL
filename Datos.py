# SE IMPORTA BASE DE DATOS

from ucimlrepo import fetch_ucirepo
# Fetch dataset from UC Irvine Machine Learning Repository
dataset = fetch_ucirepo(id=544)

# Data (as pandas dataframes)
X = dataset.data.features
y = dataset.data.targets

# Print metadata of the dataset
print("Metadata del conjunto de datos:")
print(dataset.metadata)

# Print variable information
print("\nInformación de las variables:")
print(dataset.variables)

# Print first few rows of the features and targets
print("\nPrimeras filas de las características:")
print(X.head())

print("\nPrimeras filas de los objetivos:")
print(y.head())