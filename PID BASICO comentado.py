# -*- coding: utf-8 -*-
"""
Created on Wed May  5 01:56:43 2021
@author: damian
Comandos implementados en la placa:
V###.### ó v###.###:    establece la tensión de salida del controlador.
                        valor usado por el controlador para fijar la modulación por ancho de pulso: PWM = 0,5 (1+Vsalida/Vmaxima).
R####### ó R#######:    fija la resolución del encoder.
                        solo se usa para transformar los puloss a grados pero no esta implementado.
M###.### ó m###.###:    establece la máxima tesion soportada por el motor.
                        valor usado por el controlador para fijar la modulación por ancho de pulso: PWM = 0,5 (1+Vsalida/Vmaxima).
X ó x:                  reinicia la placa.
"""
import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'qt5') #Hace que los gráficos se generen en una ventana a parte
#get_ipython().run_line_magic('matplotlib', 'inline') #Hace que los gráficos se vean en el spyder

#%% Iniciar controlador serie
ser = serial.Serial(port='COM5', baudrate=115200, bytesize=8, parity='N', stopbits=1, timeout=0.005, xonxoff=0, rtscts=0)
ser.close() 
ser.open()

#reset controlador
ser.write(bytes('X','utf-8')) 
time.sleep(0.01)
ser.flushInput()

#escribo voltaje, pregunto posicion y velocidad
str = 'V0\n\r'
ser.write(bytes(str,'utf-8'))
time.sleep(0.002)
s = ser.readline(25)
print(s)

#%% Defino funciones de comunicacion con el controlador

def setVoltageGetData(puerto,voltaje):
    puerto.flushInput()
    str = 'V%f\n\r' % (voltaje)
    puerto.write(bytes(str,'utf-8'))
    time.sleep(0.002)
    s = puerto.readline(25)
    pos = float(s[0:9])
    vel = float(s[10:23])  
    return pos,vel

def resetControlador(puerto):
    puerto.write(bytes('X','utf-8')) 
    time.sleep(0.01)
    puerto.flushInput()

def voltajeCorregido(voltaje):
    voltpos = 2
    voltneg = 2 
    maxvolt = 12
    if(voltaje > 0 ):        
        voltaje *= maxvolt/(maxvolt+voltpos)
        voltaje += voltpos
    else:        
        voltaje *= maxvolt/(maxvolt+voltneg)
        voltaje -= voltneg
    return voltaje

#%% Respuesta a un pulso de voltaje

#reseteo el controlador    
resetControlador(ser)
time.sleep(0.2)

#inicializo variables
voltajes = np.concatenate((0*np.ones(10) ,4*np.ones(50), 0*np.ones(140)))
N = len(voltajes)
posiciones = np.zeros(N)
posiciones[:] = np.nan
velocidades=np.zeros(N)
velocidades[:] = np.nan
tiempos=np.zeros(N)
tiempos[:] =np.nan

#loop poniendo voltaje y midiendo posicion y velocidad
toc = time.time()
for i in range(N):    
    pos,vel = setVoltageGetData(ser,voltajes[i]) 
    posiciones[i] = pos
    velocidades[i] = vel
    tiempos[i] = time.time()-toc


#plot de la salida
plt.close('all')
fig, axs = plt.subplots(3, sharex=True)
axs[0].plot(tiempos, voltajes,'.-')
axs[0].set(ylabel = 'Voltaje')
axs[1].plot(tiempos, posiciones,'.-')
axs[1].set(ylabel = 'Posición')
v2 = np.diff(posiciones) / np.diff(tiempos) /256
axs[2].plot(tiempos[:-1], v2,'.-')
axs[2].plot(tiempos, velocidades,'.-')
axs[2].set(ylabel = 'Velocidad')
plt.legend(('Medida por la PC','Pedida por la placa'))
plt.xlabel('Tiempo [s]')

#%%
maximo_vel = max(velocidades) # pide el maximo de la lista velocidades
plt.plot(tiempos,posiciones,".-")
plt.plot(tiempos,velocidades,".-")
#plt.plot(tiempos, maximo_vel*tiempos) #Grafica lo que deberia ser la linea tangente a la posicion de vel_maxima
#%%--------------------------------------Caracterizacion del motor----------------------------------------------
#Sirve para hacer variar el voltaje del motor y que mida la velocidad

voltajes = []
i = -15
j =-i
while i <= j:
    voltajes.append(i)
    i = i + 1

def caracterizacion(voltajes):
    N = len(voltajes) 
    velocidades = []
    for i in range(N):    
        pos,vel = setVoltageGetData(ser,voltajes[i]) #Setea el voltaje 
        time.sleep(1)                                #Espera un tiempo para que llegue a la velocidad estacionaria
        velocidades.append(vel)                      #Mete el voltaje en la lista
    vel = setVoltageGetData(ser,0)[1]                #Pone el voltaje a cero para que pare la rueda
    return velocidades[1:len(velocidades)]
    
velocidades = caracterizacion(voltajes)
plt.plot(voltajes[0:len(velocidades)],velocidades,".-")
plt.xlabel("Voltaje (V)")
plt.ylabel("Velocidad")
plt.savefig("Caracterizacion motor.png")
open("DatosCaracterizacion.txt","w")

b = np.transpose(np.array([voltajes[0:len(velocidades)],velocidades])) #Pone en dos columnas voltaje y velocidad
np.savetxt("DatosCaracterizacion.txt",b , delimiter = ",")             #Los mete en un txt

#%%--------------------------------------Control P de velocidad----------------------------------------------
def control_p(vel, K_p): #vel es el setpoint

    vel_actual = 0                  #Definimos la velocidad actual para 
    t0 = time.time()                #Definimos el tiempo inicial
    velocidades = []                #Hacemos una lista para poner las velocidades para graficarlas
    tiempo = []
    while time.time()-t0 < 1:                       #Pedimos que corra hasta cierto tiempo
        u = K_p * (vel-vel_actual)                  #Definimos u
        vel_actual = setVoltageGetData(ser, u)[1]   #Le pedimos que el voltaje sea u y que calcule la vel_actual
        velocidades.append(vel_actual)              #Mete la vel_actual en una lista
        tiempo.append(time.time()-t0)
#        print(time.time()-t0)                      #Muestra como cambia el tiempo
    setVoltageGetData(ser,0)                        #Pone el voltaje a 0
    return tiempo,velocidades

#tiempo, velocidad = control_p(300,0.1)        #Esto es para hacerlo una vez, abajo esta el barrido
#plt.figure(figsize=(10,10))              
#plt.plot(tiempo,velocidad)

lista_K_p = np.logspace(-4,-1,5)               #Crea una lista logarítmica de valores de Kp
h = 1                                          #Crea un índice 
plt.figure(figsize=(10,10))                    #Crea una figura para hacer los gráficos y ponerle cosas encima
for K_p in lista_K_p:
    t,v = control_p(300,K_p)
    plt.plot(t,v,"-",label = "K%0.0f" % (h))
    h = h + 1
    plt.xlabel("Tiempo (s)",fontsize=20)       #Le pone título al eje x y cambia el tamaño de la letra
    plt.ylabel("Velocidad",fontsize = 20)      #Le pone título al eje y y cambia el tamaño de la letra
    plt.legend(prop={"size":15}, loc = "lower right")  #Pone la leyenda, determina el tamaño y la posicion
    parameters = {'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize':30, 'axes.labelsize':30} 
    plt.rcParams.update(parameters)            #La linea de arriba y esta cambian el tamaño de los números de los ejes
    setVoltageGetData(ser,0)
    time.sleep(1)
plt.savefig("Control P barrido.png")         #Guarda el gráfico en un png

#%%--------------------------------------Control PI de velocidad----------------------------------------------
def control_pi(vel,K_p, K_i):
    vel_actual = 0
    t0 = time.time()
    velocidades = [0] #Se ponen unos 0 al principio porque sino la integral no se puede calcular
    tiempo = [0]
    
    integral = 0
    while time.time()-t0 < 4:
        u = K_p * (vel-vel_actual) + K_i* integral
        vel_actual = setVoltageGetData(ser, u)[1]
        velocidades.append(vel_actual)
        tiempo.append(time.time()-t0)            #Mete la diferencia de tiempo en una lista
        integral = integral + np.diff(tiempo[-2:])[0]*(vel-vel_actual) #Suma a si mismo la nueva parte de la integral
                                                 #np.diff(tiempo[-2:]) hace una lista con la diferencia de los ultimos
                                                 #dos tiempos y al pedirle [0] se le pide el unico elemento de la lista
    setVoltageGetData(ser,0)
    return tiempo,velocidades

#tiempo, velocidad = control_pi(300,0.01,1)      #Esto es para hacerlo una vez, abajo esta en barrido
#plt.figure(figsize=(10,10))
#plt.plot(tiempo,velocidad,'.-')

lista_K_i = np.logspace(-2,1,5)
h = 1
plt.figure(figsize=(10,10))
for K_i in lista_K_i:
    t,v = control_pi(300,0.01,K_i)
    plt.plot(t,v,"-",label = "K%d" % (h))
    h = h + 1
    plt.xlabel("Tiempo (s)",fontsize=20)
    plt.ylabel("Velocidad",fontsize = 20)
    plt.legend(prop={"size":15}, loc = "lower right")
    parameters = {'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize':30, 'axes.labelsize':30} 
    plt.rcParams.update(parameters)
    setVoltageGetData(ser,0)
    time.sleep(1)

plt.savefig("Control PI barrido 1.png") 

#%%--------------------------------------Control PID de velocidad----------------------------------------------
def control_pid(vel,K_p,K_i,K_d):
    vel_actual = 0
    t0 = time.time()
    velocidades = [0,0]
    tiempo = [0,0]
    integral = 0
    while time.time()-t0 < 5:
        u = K_p * (vel-vel_actual) + K_i* integral + K_d*((vel-vel_actual)/np.diff(tiempo[-2:])[0]) 
        vel_actual = setVoltageGetData(ser, u)[1]
        velocidades.append(vel_actual)
        tiempo.append(time.time()-t0)
        integral = integral + np.diff(tiempo[-2:])[0]*(vel-vel_actual)
    setVoltageGetData(ser,0)
    return tiempo,velocidades

tiempo, velocidad = control_pid(300,0.01,1,1)
plt.figure(figsize=(10,10))
plt.plot(tiempo,velocidad,'.-')

lista_K_d = np.logspace(-2,1,5) #Para hacer un barrido en Kd
h = 1
plt.figure(figsize=(10,10))
for K_d in lista_K_d:
    t,v = control_pid(300,0.01,1,K_d)
    plt.plot(t,v,"-",label = "K%d" % (h))
    h = h + 1
    plt.xlabel("Tiempo (s)",fontsize=20)
    plt.ylabel("Velocidad",fontsize = 20)
    plt.legend(prop={"size":15}, loc = "lower right")
    parameters = {'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize':30, 'axes.labelsize':30} 
    plt.rcParams.update(parameters)
    setVoltageGetData(ser,0)
    time.sleep(1)

plt.savefig("Control PID barrido 1.png") 
#%%

