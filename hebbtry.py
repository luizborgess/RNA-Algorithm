import numpy as np
import pandas as pd

df=pd.read_csv('Treino/dataA.csv', sep=',',header=None)
df2=pd.read_csv('Treino/dataB.csv', sep=',',header=None)
df3=df.append(df2)

print("Treinamento\n")

# Inicializa pesos zerados
w = np.zeros((1, 1025))

# Inicializa entradas

#x = np.array([[-1, -1, 1], [-1, 1, 1]])
x=df3.values
print("Entradas x = \n", x)


t = np.array([[-1, -1], [-1, 1]])
print("Alvos t = \n", t)

#PARTE DE TREINO
# Quantide de linhas para as entradas
linhas = np.shape(x)[0]

print("linhas: ",linhas)
for i in range(linhas):

    # extrai a linha i da matriz de entrada x
    xi = np.reshape(x[i], (1, np.shape(x[i])[0]))


    # extrai a linha i da matriz de alvos t
    ti = np.reshape(t[i], (1, np.shape(t[i])[0]))

    # transpõe a linha em coluna
    ti = np.transpose(ti)

    # Multiplica t^T*x
    dw = np.dot(ti, xi)
    #print("dw = \n", dw)
    w = w + dw

print("\n\nresultado w = \n", w, "\n\n")

#PARTE DE VERIFICAÇAO TESTE

def funAtivacao(a):
    if (a[0] < 0):
        return -1
    else:
        return +1
    pass


##TESTE


df4=pd.read_csv('Teste/data.csv', sep=',',header=None)
#print(df4.loc[0].to_frame().T.values)
#mude o numero de acordo com a legenda
#legend:
# 0 : A1
# 1 : A2
# 2 : A3
# 3 : A4
# 4 : B1
# 5 : B2
# 6 : B3
# 7 : B4 

v=df4.loc[0].to_frame().T.values

xiT = np.transpose(v)
net = np.dot(w, xiT)
y = np.apply_along_axis(funAtivacao, -1, net)

if y[0] == -1 and y[1]== -1:
    print('A',y)
else:
    print('B',y)
