# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 02:54:20 2020

@author: DANIEL VALLEJO
"""

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#carga del dataframe
total_data = pd.read_csv('C:/Users/DANIEL VALLEJO/Downloads/trabajofinalsenales/salidatot.csv',sep=';',decimal=",")
#agrupación de nuevos dataframes según su estado.
sanos=total_data.loc[:, "Numestdo"]==0
df_sanos=total_data.loc[sanos]

sibilancias=total_data.loc[:, "Numestdo"]==1
df_sibilancias=total_data.loc[sibilancias]

crepitancias=total_data.loc[:, "Numestdo"]==2
df_crepitancias=total_data.loc[crepitancias]

ambos=total_data.loc[:, "Numestdo"]==3
df_ambos=total_data.loc[ambos]

#print(df_crepitancias.isnull()) #verificacion de datos nulos

# print(total_data.describe())
# print(df_sanos.describe())
# print(df_sibilancias.describe())
# print(df_crepitancias.describe())
# print(df_ambos.describe())
# In[ ]:
#obtenión del promedio para cada indice del dataframe completo, y para cada estado.
df_prom_total = total_data[['Varianza','Rango','SMA grueso','Promedio del espectro','SMA fino',"Estado"]]
promedios_total=df_prom_total.groupby(["Estado"],as_index=False).mean()
print(promedios_total)

df_prom_sanos = df_sanos[['Varianza','Rango','SMA grueso','Promedio del espectro','SMA fino',"Estado"]]
promedios_sanos=df_prom_sanos.groupby(["Estado"],as_index=False).mean()
print(promedios_sanos)

df_prom_sibilancias = df_sibilancias[['Varianza','Rango','SMA grueso','Promedio del espectro','SMA fino',"Estado"]]
promedios_sibilancias=df_prom_sibilancias.groupby(["Estado"],as_index=False).mean()
print(promedios_sibilancias)

df_prom_crepitancias = df_crepitancias[['Varianza','Rango','SMA grueso','Promedio del espectro','SMA fino',"Estado"]]
promedios_crepitancias=df_prom_crepitancias.groupby(["Estado"],as_index=False).mean()
print(promedios_crepitancias)

df_prom_ambos = df_ambos[['Varianza','Rango','SMA grueso','Promedio del espectro','SMA fino',"Estado"]]
promedios_ambos=df_prom_ambos.groupby(["Estado"],as_index=False).mean()
print(promedios_ambos)
# In[ ]:
#diagrama de cajas que posibilita la visualización a simple vista de la mediana y los cuartiles de los datos,​ pudiendo también representar los valores atípicos de estos.
for i in df_prom_total.columns[0:5]:
    plt.figure(i)
    sns.boxplot(x='Estado', y=str(i), data=df_prom_total)
# In[ ]:
#ahora realizando la matriz de correlaciones para ver como se relacionan las variables estadisticas obtenidas en el data frame
#para el dataframe completo
correlation_matrix = df_prom_total.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Matriz de correlación de todo el dataframe')
plt.show()
#para el dataframe de sanos
correlation_matrix = df_prom_sanos.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Matriz de correlación para normales')
plt.show()
#para el dataframe de sibilancias
correlation_matrix = df_prom_sibilancias.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Matriz de correlación para sibilancias')
plt.show()
#para el dataframe de crepitancias
correlation_matrix = df_prom_crepitancias.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Matriz de correlación para crepitancias')
plt.show()
#para el dataframe de sibilancias y crepitancias
correlation_matrix = df_prom_ambos.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Matriz de correlación para sibilancias y crepitancias')
plt.show()
# In[ ]:
#histogramas para cada grupo con cada índice.
for i in df_prom_total.columns[0:5]:
    count,bin_edges = np.histogram(df_prom_total[str(i)])
    df_prom_total[str(i)].plot(kind='hist')
    plt.xlabel(str(i))
    plt.ylabel('Cantidad')
    plt.title('Histogramas para todo dataframe completo')
    plt.grid()
    plt.show()
    
for i in df_prom_sanos.columns[0:5]:
    count,bin_edges = np.histogram(df_sanos[str(i)])
    df_sanos[str(i)].plot(kind='hist')
    plt.xlabel(str(i))
    plt.ylabel('Cantidad')
    plt.title('Histogramas para sanos')
    plt.grid()
    plt.show()
    
for i in df_prom_sibilancias.columns[0:5]:
    count,bin_edges = np.histogram(df_prom_sibilancias[str(i)])
    df_prom_sibilancias[str(i)].plot(kind='hist')
    plt.xlabel(str(i))
    plt.ylabel('Cantidad')
    plt.title('Histogramas para sibilancias')
    plt.grid()
    plt.show()

for i in df_prom_crepitancias.columns[0:5]:
    count,bin_edges = np.histogram(df_prom_crepitancias[str(i)])
    df_prom_crepitancias[str(i)].plot(kind='hist')
    plt.xlabel(str(i))
    plt.ylabel('Cantidad')
    plt.title('Histogramas para crepitancias')
    plt.grid()
    plt.show()
    
for i in df_prom_ambos.columns[0:5]:
    count,bin_edges = np.histogram(df_prom_ambos[str(i)])
    df_prom_ambos[str(i)].plot(kind='hist')
    plt.xlabel(str(i))
    plt.ylabel('Cantidad')
    plt.title('Histogramas para crepitancias y sibilancias')
    plt.grid()
    plt.show()
    
#se observa que no hay distribución normal para ningún índice para ningún grupo. Por lo tanto es necesario realizar una prueba no paramétrica.
# In[ ]:
from scipy import stats
#se realiza la correlación de spearman puesto que se obtuvo una distribución que no es normal, como los datos están discretizados se utiliza esta correlación  y además es apta para pruebas no paramétricas
#se realiza para todo el dataframe
for i in df_prom_total.columns[0:5]:
    for j in df_prom_total.columns[0:5]:
        if not (i==j and j==i) :
            c,p = stats.spearmanr(df_prom_total[str(j)], df_prom_total[str(i)])
            print('Resultado Spearman de '+str(i)+' '+'contra'+' '+str(j)+' en dataframe completo. La correlacion es: '+ str(c) +'  y el valor p es: ' + str(p))
#para sanos
for i in df_prom_sanos.columns[0:5]:
    for j in df_prom_sanos.columns[0:5]:
        if not (i==j and j==i) :
            c,p = stats.spearmanr(df_prom_sanos[str(j)], df_prom_sanos[str(i)])
            print('Resultado Spearman de '+str(i)+' '+'contra'+' '+str(j)+'en dataframe sanos. La correlacion es: '+ str(c) +'  y el valor p es: ' + str(p))

#para sibilancias
for i in df_prom_sibilancias.columns[0:5]:
    for j in df_prom_sibilancias.columns[0:5]:
        if not (i==j and j==i) :
            c,p = stats.spearmanr(df_prom_sibilancias[str(j)], df_prom_sibilancias[str(i)])
            print('Resultado Spearman de '+str(i)+' '+'contra'+' '+str(j)+'en dataframe sibilancias. La correlacion es: '+ str(c) +'  y el valor p es: ' + str(p))
    
#para crepitancias
for i in df_prom_crepitancias.columns[0:5]:
    for j in df_prom_crepitancias.columns[0:5]:
        if not (i==j and j==i) :
            c,p = stats.spearmanr(df_prom_crepitancias[str(j)], df_prom_crepitancias[str(i)])
            print('Resultado Spearman de '+str(i)+' '+'contra'+' '+str(j)+'en dataframe cripitancias. La correlacion es: '+ str(c) +'  y el valor p es: ' + str(p))

#para ambos
for i in df_prom_ambos.columns[0:5]:
    for j in df_prom_ambos.columns[0:5]:
        if not (i==j and j==i) :
            c,p = stats.spearmanr(df_prom_ambos[str(j)], df_prom_ambos[str(i)])
            print('Resultado Spearman de '+str(i)+' '+'contra'+' '+str(j)+', en dataframe de sibilancias y crepitancias. La correlacion es: '+ str(c) +'  y el valor p es: ' + str(p))  


# In[ ]:

#ahora haciendo la prueba de domparacion entre los grupos:
#para sanos contra sibilancias
for i in df_prom_total.columns[0:5]:
    sta,p_val=stats.mannwhitneyu(df_prom_sanos[i], df_prom_sibilancias[i])
    print(i, ', sanos contra sibilancias')
    print('Prueba de Mann Whitney U para sanos contra sibilancias: Estadística:'+str(sta)+' y valor p:'+ str(p_val))

#para sanos contra crepitancias
for i in df_prom_total.columns[0:5]:
    sta,p_val=stats.mannwhitneyu(df_prom_sanos[i], df_prom_crepitancias[i])
    print(i, ', sanos contra crepitancias')
    print('Prueba de Mann Whitney U para sanos contra crepitancias: Estadística:'+str(sta)+' y valor p:'+ str(p_val))

#para sanos contra crepitancias y sibilancias
for i in df_prom_total.columns[0:5]:
    sta,p_val=stats.mannwhitneyu(df_prom_sanos[i], df_prom_ambos[i])
    print(i, ', sanos contra sibilancias y crepitancias')
    print('Prueba de Mann Whitney U para sanos contra sibilancias y crepitancias: Estadística:'+str(sta)+' y valor p:'+ str(p_val))

#para sibilancias contra crepitancias 
for i in df_prom_total.columns[0:5]:
    sta,p_val=stats.mannwhitneyu(df_prom_crepitancias[i], df_prom_sibilancias[i])
    print(i, ', crepitancias contra sibilancias')
    print('Prueba de Mann Whitney U para crepitancias contra sibilancias: Estadística:'+str(sta)+' y valor p:'+ str(p_val))