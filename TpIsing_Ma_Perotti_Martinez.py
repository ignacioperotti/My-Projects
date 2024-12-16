# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 09:42:41 2021

@author: Exactas
"""


import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.optimize import curve_fit
from numpy.random import rand, random, choice
from IPython import get_ipython


#%%

get_ipython().run_line_magic('matplotlib', 'qt5')
@njit 

def h(S):# energía por particula
    H = 0 
    L=len(S)
    for a in range(L):
        for b in range(L):
            H= -S[a, b]*(S[a-1, b] + S[a, b-1]) #conocida expresión para el Hamiltoniano
    return H


 #%%   
#Utilizamos que el incremento de la energía es 2 veces la energía inicial 



@njit
def metropolis(S,prob):
  L=len(S)
  dm=0 
  de=0
  for i in range(L**2): 
    a=np.random.randint(L) #números aleatorios ente 0 y L-1
    b=np.random.randint(L)
    s=S[a,b] #fijo un espin 
    #diferencia de energía para el espin s fijado, con condicion de cadena cerrada
    d_energia=2*s*(S[(a-1)%L,b]+S[(a+1)%L,b]+S[a,(b-1)%L]+S[a,(b+1)%L])
    #defino un R con valor entre 0 y 1 que me va a servir para definir las condiciones de energia
    R=np.random.rand()
    #Propongo las condiciones: si la dif de energia es positiva: flipea con prob, si es <=0, flipea
    if d_energia<=0:
          s*=-1
    elif d_energia==4 and R<prob[0]:
          s*=-1
    elif d_energia==8 and R<prob[1]:
          s*=-1
    #para magnetizacion resto el valor del espin s que fije y transforme, con el espin original S[a,b] 
    dm=dm+(s-S[a,b])/(L**2)
    #lo mismo para la dif de energia
    de=de+(S[a,b]-s)*(S[(a-1)%L,b]+S[(a+1)%L,b]+S[a,(b-1)%L]+S[a,(b+1)%L])/(L**2)
    #repito este proceso para todos los s
    S[a,b]=s
  return S, dm, de

'''Se conoce que el Hamiltoniano es la suma en $a,b$ de -S[a,b](S[a-1,b]+S[a+1,b]), de allí la definición dada de h(S). Se puede ver entonces cuando calculo la diferencia de energía
para un espin fijo, por ejemplo el S[1,2], puedo escribir el Hamiltoniano del sistema de la siguiente manera : -S[1,2]* (suma de primeros vecinos)-la interaccion de los demas espin.
Al pasar por la transformación (o no), el espin S[1,2] se convierte en s[1,2], pero el resto no cambió, entonces
la diferencia de energía para el elemento 1,2 es de -(s[1,2]-S[1,2])*(suma de primeros vecinos), en el caso que se da vuelta $s[1,2]-S[1,2]=-S[1,2] $ y
obtengo la expresion de d_energia. 

Por el otro lado, para hallar los de y dm conozco que M=suma de todos los espines=S[1,2]+resto, entonces dm para una partícula es 
s[1,2]-S[1,2], que lo divido por $L^2$ para normalizar. El razonamiento para de es análogo
'''

#%%

L=30 
beta=0.5*3

nequilibrio=20000 #Numero de pasos que proponemos para llegar al equilibrio

prob=np.array([np.exp(-4*beta),np.exp(-8*beta)])

#Estado inicial
#Opcion 1: todos los spines apuntan para arriba
S=choice([1, 1], size=(L, L))  
#Opcion 2: matriz aleatoria de 1's y -1's
#S=choice([1, -1], size=(L, L))  

print(metropolis(S,prob))

m=np.zeros(nequilibrio) #magnetización en función del paso
e=np.zeros(nequilibrio) #energia por particula en funcion del paso

m[0]=np.mean(S)
e[0]=h(S)

for n in range(1,nequilibrio):
  S,dm,de=metropolis(S,prob)
  m[n]=m[n-1]+dm
  e[n]=e[n-1]+de

plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
plt.plot(m)
plt.ylabel('magnetizacion')
plt.subplot(2,2,3)
plt.plot(e)
plt.ylabel('energia')
plt.xlabel('paso')
plt.subplot(1,2,2)
plt.imshow(S) #plotea el estado final, dandole un color al 1 y otro al -1
plt.show()

#%%


L=30
beta=0.5

nequilibrio=1000 #Este valor lo decidimos en base a lo anterior
npromedio=20000 #Numero de pasos tentativo para calcular promedios

prob=np.array([np.exp(-4*beta),np.exp(-8*beta)])

S=np.ones((L,L),dtype=int) #Estado inicial


for n in range(1,nequilibrio):
  S,dm,de=metropolis(S,prob) #Termalizamos

m=np.zeros(npromedio)
e=np.zeros(npromedio) #Magnetizacion y energia por particula en funcion del paso
m[0]=np.mean(S)
e[0]=h(S)

mmedia=[abs(m[0])]
emedia=[e[0]] #Valores medios en funcion del numero de pasos para promediar

for n in range(1,npromedio):
  S,dm,de=metropolis(S,prob)
  m[n]=m[n-1]+dm
  e[n]=e[n-1]+de
  mmedia.append(np.mean(abs(m[0:n+1])))
  emedia.append(np.mean(e[0:n+1]))

plt.figure()
plt.subplot(2,1,1)
plt.plot(mmedia)
plt.ylabel('magnet media')
plt.subplot(2,1,2)
plt.plot(emedia)
plt.ylabel('energia media')
plt.xlabel('número de pasos')
plt.show()

#%%


L=30 
beta=np.linspace(0.1,10,30)

nequilibrio=1000 #Numero de pasos que proponemos para llegar al equilibrio
npromedio=20000

prob=np.array([np.exp(-4*beta),np.exp(-8*beta)])

#Estado inicial
#Opcion 1: todos los spines apuntan para arriba
S=choice([1, 1], size=(L, L))  
#Opcion 2: matriz aleatoria de 1's y -1's
#S=choice([1, -1], size=(L, L))  

#Definimos unos ceros y luego voy creando una lista usando un loop
pbetas=len(beta)
Mmedia=np.zeros(pbetas)
Emedia=np.zeros(pbetas)
C=np.zeros(pbetas)
Chi=np.zeros(pbetas)

for p in range(pbetas):    
    if beta[p]>0.4 and beta[p]<0.5:
      npromedio=40000
    else:
      npromedio=20000
      
      #incorporo la termalizacion para estas betas 
      for n in range(1,nequilibrio):
          S,dm,de=metropolis(S,prob) #Termalizamos

      m=np.zeros(npromedio)
      e=np.zeros(npromedio) #Magnetizacion y energia por particula en funcion del paso
      evar=[np.var(e)]#defino las varianzas de estas listas para calcular C y Xi despues
      mvar=[np.var(m)]
    
      m[0]=np.mean(S)
      e[0]=h(S)
        
      mmedia=[abs(m[0])]
      emedia=[e[0]] #Valores medios en funcion del numero de pasos para promediar
      
      for n in range(1,npromedio):
          S,dm,de=metropolis(S,prob)
          m[n]=m[n-1]+dm
          e[n]=e[n-1]+de
          mmedia.append(np.mean(abs(m[0:n+1])))
          emedia.append(np.mean(e[0:n+1]))
          evar.append(np.var(e[0:n+1]))
          mvar.append(np.var(abs(m[0:n+1])))


    Mmedia[p]=np.mean(mmedia)
    Emedia[p]=np.mean(emedia)
    C[p]=(np.mean(evar)*beta[p]**2)/(L**2)
    Chi[p]=(np.mean(mvar)*beta[p])/(L**2)


T=1/beta

plt.figure()
plt.subplot()
plt.plot(T,Emedia)
plt.ylabel('Energia media')
plt.xlabel('Temperatura')
plt.subplot()
plt.plot(T,Mmedia,)
plt.ylabel('Magnetizacion media')
plt.xlabel('Temperatura')
plt.subplot()
plt.plot(T,C)
plt.ylabel('Calor especifico')
plt.xlabel('Temperatura')
plt.subplot()
plt.plot(T,Chi)
plt.ylabel('Cusceptibilidad')
plt.xlabel('Temperatura')
plt.show()

'''Se conoce que la derivda de U en función de beta es igual a menos la varianza de energía
por como se define U apartir del ensamble canónico. Pero esta tambien es Cv/beta^2 (tomando k=1)
Entonces para energía normalizada se cumple que varianza(E)*beta^2/L^2=Cv. El proceso de susceptibidad
es análogo.'''

#%%


'''Por definición de la funcion de correlación dada en el enunciado puedo  separarla
en dos sumas, una parte del espin S[a,b]=s con sus vecinos a la izquierda y derecha
 y otra parte del resto. Entonces cuando transformo s (o no) y resto la diferencia entre 
 la función de correlación c(s)-c(S[a,b]) lo que obtengo es (s-S[a,b])*(vecinos a la izquierda y derecha)
 ,y lo divido por L^2 para normalizar.
'''

#%%

L=30

nequilibrio=1000 
npromedio=50000

l=int(L/2) #numero de componentes del vector c
m=np.zeros(npromedio) #magnetizacion en funcion del paso
corr=np.zeros((l,npromedio)) #vector c en funcion del paso

erres=np.arange(l) #dominio de la funcion de correlacion
def fitcor(r,a,xi): #funcion con la que vamos a ajustar
  return a*np.exp(-r/xi)

S=np.ones((L,L),dtype=int)

print('Longitud de correlacion:')

for T in [1.5,2.3,3]:

  beta=1/T

  prob=np.array([np.exp(-4*beta),np.exp(-8*beta)])

  #Termalizamos con la funcion metropolis(S,prob), que es mas rapida
  for n in range(1,nequilibrio):
    S,dm,de=metropolis(S,prob)

  m[0]=np.mean(S)
  corr[:,0]=cor(S)

  for n in range(1,npromedio):
    S,dm,dcorr=metropolis2(S,prob)
    m[n]=m[n-1]+dm
    corr[:,n]=corr[:,n-1]+dcorr

  mmedia=np.mean(abs(m))
  correlacion=np.mean(corr,axis=1)-mmedia**2 #funcion de correlacion

  #Ajustamos:
  parametros,covarianza=curve_fit(fitcor,erres,correlacion)
  #(la funcion curve_fit devuelve el mejor valor de los parametros
  #y tambien la covarianza, que no vamos a usar)

  print('T={} => {}'.format(T,parametros[1]))

plt.figure()
plt.plot(erres,fitcor(erres,parametros[0],parametros[1]),label='T={}'.format(T))
  
plt.xlabel('r')
plt.ylabel('funcion de correlacion')
plt.legend()
plt.show()

'''Acá nos dio mal la xi, seguramente esta mal definida la función de correlación
pero probamos un par de cosas mas y tampoco funcionaba :C'''


#%%


L=30

nequilibrio=1000 
npromedio=2000
Ts=np.linspace(1,5,10)
ListXi=[]
#Utilizando la misma funcion de antes pero ahora con mas temperaturas

for T in Ts: 

  beta=1/T
  
  prob=np.array([np.exp(-4*beta),np.exp(-8*beta)])

  #Termalizamos con la funcion metropolis(S,prob), que es mas rapida
  for n in range(nequilibrio):
    S,dm,de = metropolis(S,prob)

  m[0] = np.mean(S)
  corr[:,0] = cor(S)

  for n in range(1,npromedio):
    S,dm,dc = metropolis2(S,prob)
    m[n] = m[n-1]+dm
    corr[:,n] = corr[:,n-1]+dcorr

  mmedia = np.mean(abs(m))
  correlacion = np.mean(corr,axis=1)-mmedia**2 #funcion de correlacion

  #Ajustamos:
  parametros,covarianza = curve_fit(fitcor,erres,correlacion)
  ListXi.append(parametros[1])

plt.figure()
plt.plot(Ts,ListXi) 
plt.ylabel('Longitud de correlacion')
plt.xlabel('Temperatura')
plt.show()

'''Esto se parece a la m en funcion de temp, seguramente hubo algo raro cuando
definimos la funcion de correlación '''

#%%

L=15 #Lado de la red
T=np.linspace(2.2,2.4,30)
prob=np.array([np.exp(-4*beta),np.exp(-8*beta)])

#Estado inicial
#Opcion 1: todos los spines apuntan para arriba
S=choice([1, 1], size=(L, L))  
#Opcion 2: matriz aleatoria de 1's y -1's
#S=choice([1, -1], size=(L, L))  

nequilibrio=1000 #Numero de pasos que proponemos para llegar al equilibrio
npromedio=20000

'''Podríamos graficar diferentes C(T) con sus L, y buscar el valor de T donde el calor especifico es máximo.
De esta manera obtengo los Tc para para diferentes L y podemos graficarlo para hallar a, pero tira error las celdas de
calor específico como que se confunde en cuando arranca el loop de termalización
y no entendemos por qué'''



'''
Discusion: Entendemos que la idea de este tp es ver cómo las funciones como la Cv y Chi
son continuas cerca de Tc por ser un resultado exacto calculado por la simulación.
También se debería aproximar mejor la Tc en comparación con el caso de campo medio y campo de Bethe
donde son buenas aproximaciones para dimensiones más grandes, pero para 2D por ejemplo
presentan cantidades macroscópicas como E,M,Cv y Chi descontinuas cerca de Tc. El campo
es la aproximación que más se usó por despreciar la interaccion de un espin con sus próximos 
vecinos así que más discontinuidad tiene (por ejemplo, E=0 para T>Tc), mientras que el campo de
Bethe es mejor que la aprox de campo medio porque toma un valor medio para cada espin
con sus PV pero aín así presenta mucha discontinuidad en dimensiones bajas. Est0 debería
dar todos valores muy lindos pero no lo llegamos a ver bien
'''