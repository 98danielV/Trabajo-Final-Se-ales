# -*- coding: utf-8 -*-
"""
Created on Sun May 31 19:52:11 2020

@author: DANIEL VALLEJO
"""
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from linearFIR import filter_design, mfreqz
import scipy.signal as signal
import pywt
import scipy.io as sio;
import pandas as pd
def textos_y_audios(ruta_audio_y_textos): #funcion que me retorna listas con las señales cargada, filtradas, sr, archivo txt, etc.
    lista_archivos=os.listdir(ruta_audio_y_textos)# carga los archivos en la consola para mirarlos y los almacena en una lista.
    lista = []
    y_vacia = []
    sr_vacia = []
    y_filtrada = []
    x_ft = []
    txt =[]
    for i in np.arange(920,1840,2): #ciclo for para cargar los audios y los txt de la base de datos cargada.
        filename = ruta_audio_y_textos + '/' + lista_archivos[i+1] #carga de audio
        filetxt =  ruta_audio_y_textos + '/' + lista_archivos[i] #carga de texto
        lista.append(filename) #agrega un audio a la lista que almacena todos los audios.
        txt.append(filetxt) #agrega un txt a la lista que almacena todos los audios.
    for j in np.arange(0,len(lista)): #ciclo for para cargar los audios
        y, sr = librosa.load(lista[j]) #obtiene la señal de audio con sus valores y la frecuencia de muestreo.
       
        y_vacia.append(y)#agrega los valores de cada señal de audio a la lista
        sr_vacia.append(sr)

    for m in np.arange(0,len(y_vacia)): #ciclo for que ejecuta la cantidad de veces de audios cargados.
        #diseño de filtros paso bajo y alto de la señal para eliminar frecuencias o ruido indeseado.
        order, lowpass = filter_design(22050, locutoff = 0, hicutoff = 1000, revfilt = 0);
        order, highpass = filter_design(22050, locutoff = 100, hicutoff = 0, revfilt = 1);
        y_hp = signal.filtfilt(highpass, 1, y_vacia[m]);
        y_bp = signal.filtfilt(lowpass, 1, y_hp);
        y_bp1 = np.asfortranarray(y_bp)
        #despues del filtrado de cada señal se agrega a la lista que contiene todas la señales de audio filtradas con las especificaciones de los filtros creados (pasa bajas y altas)
        #y luego se aplica el comanado filt filt para realizar el filtrado de cada señal.
        y_filtrada.append(y_bp1);

    for g in np.arange(0,len(y_filtrada)): #cilco for que se ejecute la cantidad de audios de la base de datos, 
        #definicion de funciones que se necesitan para realizar el filtrado wavelet 
        def wthresh(coeff,thr):
            y   = list();
            s = wnoisest(coeff);
            for i in range(0,len(coeff)):
                y.append(np.multiply(coeff[i],np.abs(coeff[i])>(thr*s[i])));
            return y;
    
        def thselect(y_bp1):
            Num_samples = 0;
            for i in range(0,len(y_bp1)):
                Num_samples = Num_samples + y_bp1[i].shape[0];
    
            thr = np.sqrt(2*(np.log(Num_samples)))
            return thr

        def wnoisest(coeff):
            stdc = np.zeros((len(coeff),1));
            for i in range(1,len(coeff)):
                stdc[i] = (np.median(np.absolute(coeff[i])))/0.6745;
            return stdc;
        
        plt.figure(g)
        LL = int(np.floor(np.log2(y_filtrada[g].shape[0])));
    
        coeff = pywt.wavedec( y_filtrada[g], 'db6', level=LL );

        thr = thselect(coeff);
        coeff_t = wthresh(coeff,thr);
        
        x_rec = pywt.waverec( coeff_t, 'db6');
        
        x_rec = x_rec[0:y_filtrada[g].shape[0]];
       #plt.plot(y_vacia[i][0:5000],label='Original')
       #plt.plot(x_rec[0:5000],label='Umbralizada por Wavelet y aplicada filtros ')
        
        x_filt = np.squeeze(y_filtrada[g] - x_rec); #resta de la señal con filtros pasabajo y pasa altos menos esa misma señal pero aplicado wavelet para obtner las señales respiratorias
        x_ft.append(x_filt)
        #plt.plot(x_filt[0:5000],label='Original - Umbralizada')
        #plt.legend(loc="upper right")
    return y_vacia,x_ft,sr_vacia,lista,txt      
            
#textos_y_audios("C:/Users/JUAN CRUZ/Downloads/audio_and_txt_files")       

def cyclesanotation(rutawav,rutatxt,op): #funcion que recibe los archivos de audio fikltrado y texto
    
    archivo = np.loadtxt(rutatxt)
    if op==0:
        audio,sr = librosa.load(rutawav)
    if op==1:
        audio,sr = rutawav
    #ime = audio/sr
    timen= []
    silibancias = []
    crepitancias = []
    anormals=[]
    anormalc = []
    anormal = []
    normal = []
    tnormal = []
   
    for i in archivo: #para cada archivo carga los tiempos y mira si tiene crepitancias o sibilacias
   
        ti = i[0]
        tf = i[1]
        inicio =int(np.round(ti*sr))
        fin = int(np.round(tf*sr))
        
        if i[3] ==0  and i[2]==0:
            timen.append(i[0:2])
            normal.append(audio[inicio:fin])
            print('No tiene silibancias o crepitancias')
        elif i[3]==1 and i[2]==1:
            tnormal.append(i[0:2]) #agrega los valore normales a la lista.
            anormal.append(audio[inicio:fin]) #tiempo en ele que se producen estas oscilaciones y se agrega a una lista
            
            
        elif i[2]==1 and i[3]==0:
            crepitancias.append(i[0:2]) #agrega las crepitancias a la lista
           
            anormalc.append(audio[inicio:fin]) #tiempo en ele que se producen estas perturbacione sy lo agrega a una lista
         
        elif i[3]==1 and i[2]==0:
            silibancias.append(i[0:2]) #agrega las sibilancias a la lista
            anormals.append(audio[inicio:fin]) #tiempo en ele que se producen estas perturbacione sy lo agrega a una lista

        else:
            print('caracter diferente de 1 o 0')
    
    return silibancias,crepitancias,anormals,anormalc,timen,normal,tnormal,anormal #retorno de las listas
#PROCESAMIENTO Y EXTRACCION CARACTERISTICAS
#preprocesado
#averagerespiratorycycle = 15
#audio signals sampled 44100hz
#abnormal sounds 50-2400 hz
#4096  samples or 09s ms audio
#131072 samples per window
#300 windows for crackles

#variance

#range
#The simplest features of our feature set, 
#is the maximum value of the audio ﬁle subtracted 
#from the minimum value of the audio ﬁle. 
#R(S) = [Max(s)-Min(s)];S = signal

#Sum of simple moving average
#SMAC (SIG) = SUM(SIG(N-1)-SIGN),(N=1,len(SIG))
#SMAFINE (SIG)=MAX(SMAC(window1),SMAC(window2),....)



#f(x) = sign{F(x)}
#where
#   F(x) =
#   K X k=1
#   akΦ(x)

#where Φ(x)is the base classiﬁer,
#which returns a binary class label. 
#   K denotes the number of classiﬁers being boosted,
#and a is the weight associated with the kth weak classiﬁer.
#Finding the α is done through iterative, or stepwise,
#optimization of m steps, where Fm−1(x) is the previous
# optimized iteration. 


#am = argminJ(a)

#J(a) =N X i=1 exp(−yi(Fm−1(xi)+ aΦ(xi)))

#Support vector machine
def indcycle(rcycle,sr): #funcion que saca los valores estadisticos como rango y varianza de cada señal, así como también su densidad despectral de potencia
    var = np.var(rcycle)
    rango = np.abs(np.max(rcycle)-np.min(rcycle))
    #SMAC (SIG) = SUM(SIG(N-1)-SIGN),(N=1,len(SIG))
    #SMAFINE (SIG)=MAX(SMAC(window1),SMAC(window2),....)

    smac = 0
    for j in range(1,len(rcycle)):
        smac += np.abs(rcycle[j-1]-rcycle[j])
    f, Pxx =signal.welch(rcycle,sr,'hamming',1024,scaling='density')
    meanespec = np.mean(Pxx)
    smafine = 0
    Fine = []
    for i in range(0,len(rcycle),100):
        ventana = rcycle[i:i+800]
        for j in range(1,len(ventana)):
            smafine +=np.abs(ventana[j-1]-ventana[j])
        Fine.append(smafine)
        smafine = 0
    smafine = max(Fine)
                         
    return var,rango,smac,meanespec,smafine
    
    
    
def rutina(): #funcion general que llama a las demas funciones
    y,x,s,l,t= textos_y_audios("C:/Users/DANIEL VALLEJO/Downloads/trabajofinalsenales/Respiratory_Sound_Database/audio_and_txt_files")  
    datas =[] #alamacena los dataframes pque se crean de la base de datos

    ciclos = []
    for i in range(len(x)): #para cada señal filtrada se llama a la funcion cyclesanotation que entrega las sibilancias, crepitacias, etc, de toda la base de datos.
            
            
            ts,tc,ans,anc,tn,n,ta,an = cyclesanotation([x[i],s[i]],t[i],1) #obtencion de los parametros mencionados para cada señal
            ciclos.append(n)
            frame = {}
            #frame['Nombre']=t[i]
            cycle  = []
            health =[]
            vt = []
            rt = []
            smt = []
            mst = []
            fnt = []
            estado = []
          
            #los ciclos siguientes llaman a la funcion indcycles teneindo como argumentos lsa listas retornadas por la anterior funcion (cyclesanotation), de esta forma se pueden obtener los parámetros como varianza y rango para cada condción teniendo en cuenta todas las señales ya filtradas.
            for j in range(len(an)):
           
                v,r,sm,ms,fn=indcycle(an[j],s[i])
                health.append('Sibilancia y Crepitancia')
                estado.append(3)
                cycle.append(str(ta[j][0])+'-'+str(ta[j][1]))
                vt.append(v)
                rt.append(r)
                smt.append(sm)
                mst.append(ms)
                fnt.append(fn)
            
            for j in range(len(ans)):
           
                v,r,sm,ms,fn=indcycle(ans[j],s[i])
                health.append('Sibilancia')
                estado.append(1)
                cycle.append(str(ts[j][0])+'-'+str(ts[j][1]))
                vt.append(v)
                rt.append(r)
                smt.append(sm)
                mst.append(ms)
                fnt.append(fn)
            
                

            for k in range(len(anc)):
                
                v,r,sm,ms,fn=indcycle(anc[k],s[i])
                health.append('Crepitancia')
                cycle.append(str(tc[k][0])+'-'+str(tc[k][1]))
                estado.append(2)
                vt.append(v)
                rt.append(r)
                smt.append(sm)
                mst.append(ms)
                fnt.append(fn)
                
            for l in range(len(n)):
                v,r,sm,ms,fn=indcycle(n[l],s[i])
                health.append('Comun')
                estado.append(0)
                cycle.append(str(tn[l][0])+'-'+str(tn[l][1]))
                vt.append(v)
                rt.append(r)
                smt.append(sm)
                mst.append(ms)
                fnt.append(fn)
                
           #rs,rc,rns,rnc,rtn,rn = cyclesanotation([y[i],s[i]],t[i],1)
           #creacion de los nombres de cada columna del dataframe
            frame['Numestdo'] = np.array(estado)
            frame['Ciclo'] = np.array(cycle)
            frame['Estado']  = np.array(health)
            frame['Varianza'] = np.array(vt)
            frame['Rango']=np.array(rt)
            frame['SMA grueso'] = np.array(smt)
            frame['Promedio del espectro'] = np.array(mst)
            frame['SMA fino'] =  np.array(fnt)
            frame =pd.DataFrame(dict([(k,pd.Series(v)) for k,v in frame.items()]))
            datas.append(frame)
    #DATAFRAME
    return datas
    
d = rutina()
total = pd.concat(d) #concatena los dataframes obtenidos para toda la base de datos
total.to_csv('salida.csv',mode='a',index = False,sep = ';',decimal=',') #convierte ese dataframe total en un archivo csv



