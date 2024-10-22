import pandas as pd
import random
from autocorrect import Speller
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#Wechselkurs von Dollar zu Euro. Wechselkurs vom 03.12.2021
exchange_rate = 0.88

# DataFrame laden
df = pd.read_csv('uncleaned2_bike_sales.csv')

# Überprüfen der Kopfzeile nach Dollar-Zeichen
dollar_columns = []
for column in df.columns:
    if '$' in column:  # Überprüfen, ob Dollar-Zeichen in der Kopfzeile der Spalte enthalten ist
        dollar_columns.append(column)

print("Hier stehen die Features mnit $ Zeichen:")
print(dollar_columns)

# Umwandeln der Werte in den Spalten mit Dollar-Zeichen in Euro
for column in dollar_columns:
    df[column] = df[column] * exchange_rate



# Anzeigen der ersten paar Zeilen des umgewandelten DataFrames
print("Erste paar Zeilen des umgewandelten DataFrames:")
print(df.head())

# Überblick über die Datentypen und die Anzahl der Nicht-Null-Einträge
print("\nÜberblick über die Datentypen und Nicht-Null-Einträge:")
print(df.info())

# Schritt 2: Data Cleaning
# Auffinden von Spalten mit mehr als 60% fehlenden Einträgen
missing_threshold = len(df) * 0.6
columns_to_drop = df.columns[df.isnull().sum() > missing_threshold]

# Löschen dieser Spalten
df_cleaned = df.drop(columns=columns_to_drop)

# Auffinden von Zeilen mit mehr als 60% fehlenden Einträgen
rows_to_drop = df_cleaned[df_cleaned.isnull().sum(axis=1) > missing_threshold].index

# Löschen dieser Zeilen
df_cleaned = df_cleaned.drop(index=rows_to_drop)

df_cleaned = df_cleaned.rename(columns = {" Profit_$ ": "Profit_€", " Unit_Cost_$" : "Unit_Cost_€", " Unit_Price_$ " : "Unit_Price_€", " Cost_$" : "Cost_€", "Revenue_$": "Revenue_€" })



#Schritt 3.
# Stratified replacement für Spalte 2
# Fehlende Werte in Spalte 2 werden durch die nächsthöhere Zahl ersetzt
for i in range(len(df_cleaned)):
    if pd.isnull(df_cleaned.iloc[i, 1]):  # Überprüfen, ob der Wert in Spalte 2 fehlt
        df_cleaned.iloc[i, 1] = df_cleaned.iloc[i-1, 1] + 1  # Ersetzen des fehlenden Werts durch die nächsthöhere Zahl

# Datumsersetzung für Spalte 3 und 4
for i in range(len(df_cleaned)):
    if pd.isnull(df_cleaned.iloc[i, 2]):  # Überprüfen, ob das Datum in Spalte 3 fehlt
        df_cleaned.iloc[i, 2] = df_cleaned.iloc[i-1, 2]  # Ersetzen des fehlenden Datums durch das Datum aus der vorherigen Zeile
    if pd.isnull(df_cleaned.iloc[i, 3]):  # Überprüfen, ob der Tag in Spalte 4 fehlt
        # Ersetzen des Tags aus dem Datum der vorherigen Zeile und Zuweisen in Spalte 4
        df_cleaned.iloc[i, 3] = df_cleaned.iloc[i-1, 3]

# Ergenzen des Monats in Spalte 5
for i in range(len(df_cleaned)):
    if pd.isnull(df_cleaned.iloc[i, 4]): # Überprüfen, ob der Monat in Spalte 5 fehlt
        df_cleaned.iloc[i, 4] = df_cleaned.iloc[i-1, 4] #Ersetzen des fehlenden Monats mit dem Monat aus der vorherigen Zeile

# Ergenzen des Jahres in Spalte 6
for i in range(len(df_cleaned)):
    if pd.isnull(df_cleaned.iloc[i, 5]): # Überprüfen, ob des Jahres in Spalte 6 fehlt
        df_cleaned.iloc[i, 5] = df_cleaned.iloc[i-1, 5] #Ersetzen des fehlenden Jahres mit dem Jahr aus der vorherigen Zeile

#Fill with mean für Spalte 7 - Alter der Kunden
mean_column_7 = df_cleaned.iloc[:, 6].mean()

#Ersetzen fehlender Einträge in Spalte 7 durch den Mittelwert
for i in range(len(df_cleaned)):
    if pd.isnull(df_cleaned.iloc[i, 6]):  # Überprüfen, ob der Eintrag in Spalte 7 fehlt
        df_cleaned.iloc[i, 6] = mean_column_7  # Ersetzen des fehlenden Eintrags durch den Mittelwert



# Füllen von Spalte 8 (Customer_Gender) mit random replacement
for i in range(len(df_cleaned)):
    if pd.isnull(df_cleaned.iloc[i, 7]):  # Überprüfen, ob der Eintrag in Spalte 8 fehlt
        # Zufällig 'M' oder 'F' auswählen
        random_gender = random.choice(['M', 'F'])
        df_cleaned.iloc[i, 7] = random_gender


# Mapping von State zu Country erstellen
state_to_country = df_cleaned.dropna().set_index('State')['Country'].to_dict()

# Fehlende Einträge ergänzen
df_cleaned['Country'] = df.apply(lambda row: state_to_country.get(row['State'], row['Country']), axis=1)



# Produktkategorie ergänzen
column_name = 'Product_Category'
# Modus für Produktkategorie berechnen
mode_product_category = df_cleaned[column_name].mode()[0]
# Fehlende Werte in Produktkategorie ersetzen

df_cleaned.loc[:, column_name] = df_cleaned[column_name].fillna(mode_product_category)


# Unterkategorie ergänzen
column_name = 'Sub_Category'
# Modus für Unterkategorie berechnen
mode_sub_category = df_cleaned[column_name].mode()[0]

# Fehlende Werte in Unterkategorie ersetzen
df_cleaned.loc[:, column_name] = df_cleaned[column_name].fillna(mode_sub_category)



# Auffüllen der übrigen fehlenden Werte mit Vorwärts-/Rückwärtsfüllen
df_cleaned.ffill(inplace=True)
df_cleaned.bfill(inplace=True)



#Typos beheben
spell = Speller(lang="en")
# Konvertieren der "Month"-Spalte in Zeichenfolgen und Anwendung des Spellers
df_cleaned["Month"] = df_cleaned["Month"].apply(lambda x: spell(x))

# Entfernen von Leerzeichen am Anfang und Ende jeder Zeichenkette und mehrfache Leerzeichen innerhalb der Zeichenkette in der Spalte "Country"
df_cleaned["Country"] = df["Country"].str.strip().str.split().str.join(' ')


# Erstellen von Boxplots für jedes Feature
for column in df_cleaned.columns:
    if df_cleaned[column].dtype in ['int64', 'float64']:  # Nur numerische Spalten berücksichtigen
        plt.figure(figsize=(8, 6))
        df_cleaned.boxplot(column=[column])
        plt.title(f'Boxplot für {column}')
        plt.ylabel('Wert')
        plt.show()

# Entfernen der Ausreißer
for column in df_cleaned.columns:
    if df_cleaned[column].dtype in ['int64', 'float64']:  # Nur numerische Spalten berücksichtigen
        Q1 = df_cleaned[column].quantile(0.25)
        Q3 = df_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1
        #Standart Wert aus dem Internet war 1.5 aber der hat zu viele Entfernt daher auf 2 angehoben
        lower_bound = Q1 - 2 * IQR
        upper_bound = Q3 + 2 * IQR

        # Entfernen der Ausreißer
        df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]

# Überprüfung der Daten nach der Ausreißerentfernung
for column in df_cleaned.columns:
    if df_cleaned[column].dtype in ['int64', 'float64']:  # Nur numerische Spalten berücksichtigen
        plt.figure(figsize=(8, 6))
        df_cleaned.boxplot(column=[column])
        plt.title(f'Boxplot für {column} nach der Ausreißerentfernung')
        plt.ylabel('Wert')
        plt.show()


# Überprüfen der bereinigten Daten
print("\nBereinigte Daten:")
print(df_cleaned.head())


# Schreiben des bereinigten DataFrames zurück in die Excel-Datei, um die ursprüngliche Datei zu überschreiben
df_cleaned.to_csv('bike_sales_clean.csv', index=False)

# Zählen der Anzahl von Männern und Frauen, die ein Fahrrad gekauft haben
gender_counts = df_cleaned['Customer_Gender'].value_counts()

# Erstellen des Balkendiagramms
plt.figure(figsize=(8, 6))
gender_counts.plot(kind='bar', color=['blue', 'pink'])
plt.title('Anzahl der Fahrradkäufe nach Geschlecht')
plt.xlabel('Geschlecht')
plt.ylabel('Anzahl der Käufe')
plt.xticks(rotation=0)
plt.show()

# Gruppieren nach Land und Berechnen des durchschnittlichen Gewinns pro Land
profit_per_country = df_cleaned.groupby('Country')['Profit_€'].mean().sort_values()

# Erstellen des Balkendiagramms
plt.figure(figsize=(10, 6))
profit_per_country.plot(kind='bar', color='skyblue')
plt.title('Durchschnittlicher Gewinn pro Land')
plt.xlabel('Land')
plt.ylabel('Durchschnittlicher Gewinn')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()  # Optimierung der Layout-Anpassung
plt.show()

# Erstellen eines Streudiagramms für Männer und Frauen getrennt nach Alter und Umsatz
plt.figure(figsize=(10, 6))

# Streudiagramm für Männer
plt.scatter(df_cleaned[df_cleaned['Customer_Gender'] == 'M']['Customer_Age'],
            df_cleaned[df_cleaned['Customer_Gender'] == 'M']['Revenue_€'],
            color='blue', label='Männer')

# Streudiagramm für Frauen
plt.scatter(df_cleaned[df_cleaned['Customer_Gender'] == 'F']['Customer_Age'],
            df_cleaned[df_cleaned['Customer_Gender'] == 'F']['Revenue_€'],
            color='pink', label='Frauen')

plt.title('Ausgaben nach Alter und Geschlecht')
plt.xlabel('Alter')
plt.ylabel('Umsatz')
plt.legend()
plt.grid(True)
plt.show()

# Spalten 3, 4 und 5 löschen (Indizes 2, 3 und 4)
df_cleaned.drop(df_cleaned.columns[[0, 3, 4, 5]], axis=1, inplace=True)


# Nicht-numerische Features in numerische Features
df_cleaned["Customer_Gender"] = df_cleaned["Customer_Gender"].map({"M": 0, "F": 1})
df_cleaned["Country"] = df_cleaned["Country"].map({"United Kingdom": 0, "United States": 1, "Australia":2, "Canada":3,"France":3, "Germany":4})
df_cleaned["State"] = df_cleaned["State"].map({"England": 0, "Washington": 1, "Queensland":2, "California":3,"British Columbia":3,
                                               "New South Wales":4, "Seine (Paris)":5,"Victoria":6,"Oregon":7,"Hessen":8,"Somme":9,
                                               "Nord":10, "Nordrhein-Westfalen": 11, "Seine et Marne": 12, "South Australia":13})


df_cleaned["Product_Category"] = 1
df_cleaned["Sub_Category"] = 1

# Ersetzen des "-" durch "." in der Spalte "Date"
df_cleaned[df_cleaned.columns[1]] = df_cleaned[df_cleaned.columns[1]].str.replace("-", "")

# Löschen aller Zeilen, in denen sich kein Wert befindet
df_cleaned.dropna(inplace=True)

# Speichern des codierten Datensatzes als CSV-Datei
df_cleaned.to_csv("bike_sales_codified.csv", index=False)


# Berechnen der Korrelationsmatrix
correlation_matrix = df_cleaned.corr()

# Finden redundanter Features (mit Korrelation > 0.9)
redundant_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            redundant_features.add(correlation_matrix.columns[i])

# Löschen der redundanten Features
df_reduced = df_cleaned.drop(columns=redundant_features)

# Anzeigen der Korrelationsmatrix
print("Korrelationsmatrix:")
print(correlation_matrix)

# Standardisieren der Daten
scaler = StandardScaler()
df_standardized_no_missing = scaler.fit_transform(df_reduced)

# Durchführung der PCA
pca = PCA()
pca.fit(df_standardized_no_missing)

# Berechnen der kumulativen erklärten Varianz
explained_variance_ratio_cumulative = np.cumsum(pca.explained_variance_ratio_)

# Bestimmen der Anzahl der Hauptkomponenten, die 95% der Varianz erklären
n_components_95_variance = np.argmax(explained_variance_ratio_cumulative >= 0.95) + 1

# Anzeigen der Anzahl der Principal Components, die 95% der Varianz der Daten abdecken
print("Anzahl der Principal Components, die 95% der Varianz der Daten abdecken:", n_components_95_variance)
# Eigenwerte (Varianzen)
print("Eigenwerte (Varianzen):")
print(pca.explained_variance_)

# Eigenvektoren (Hauptkomponenten)
print("\nEigenvektoren (Hauptkomponenten):")
print(pca.components_)



# Speichern des reduzierten Datensatzes als CSV-Datei
df_reduced.to_csv('bike_sales_reduced.csv', index=False)


# Normalisieren aller Features
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df_reduced)

# Erstellen eines DataFrames aus den normalisierten Daten
df_normalized = pd.DataFrame(df_normalized, columns=df_reduced.columns)

# Speichern des normalisierten Datensatzes als CSV-Datei
df_normalized.to_csv('bike_sales_normalized.csv', index=False)