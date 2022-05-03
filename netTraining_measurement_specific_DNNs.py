#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:21:34 2019

@author: dominik
"""

# ~ import os
# ~ os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# ~ os.environ["CUDA_VISIBLE_DEVICES"] = ""

from numpy import *
from numpy import linalg as la
from scipy import linalg as sa

import keras
from keras import optimizers
from keras import initializers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Conv1D, Flatten, Reshape, AveragePooling1D,UpSampling1D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils import plot_model


def randomHaarState(dim,rank):
	A = random.normal(0,1,(dim,dim))+1j*random.normal(0,1,(dim,dim))
	q,r = la.qr(A,mode='complete')
	r  = divide(diagonal(r),abs(diagonal(r)))*identity(dim)
	rU = q@r
	B = random.normal(0,1,(dim,rank))+1j*random.normal(0,1,(dim,rank))
	B = B@B.T.conj()
	rho = (identity(dim)+rU)@B@(identity(dim)+rU.T.conj())
	return rho/trace(rho)
	
def randompure(dim,n):
    rpure = random.normal(0,1,[dim,n]) + 1j*random.normal(0,1,[dim,n])
    rpure = rpure/la.norm(rpure,axis=0)
    rhon = array([dot(rpure[:,[i]],rpure[:,[i]].conjugate().transpose())  for i in range(n)])
#     rhon = reshape(rhon,[n,4])
    return rhon
	
def stokesFromRho(rho,A):
	l = shape(A)[0]
	return array([ real(trace(rho@A[n])) for n in range(l) ])

def rhoFromStokes(stokes,A):
    global dim
    l = shape(A)[0]
    return 1/dim*(identity(dim)+sum([stokes[i]*A[i] for i in range(l)],axis=0))

def rhoFromStokesG(stokes,A):
    global dim
    l = shape(A)[0]
    return (1/dim*sum([stokes[i]*A[i] for i in range(l)],axis=0))

def pauli():
    s = zeros([3,2,2]) +1j*zeros([3,2,2])
#     s[0] = np.array([[1, 0],[0, 1]])
    s[0] = array([[1, 0],[0, -1]])
    s[1] = array([[0, 1],[ 1, 0]])
    s[2] = array([[0, -1j],[1j, 0]])
    return s

def probdists(stav,povm):
    l = shape(povm)[0]
    probtrue = array([real(trace(stav@povm[i])) for i in range(l)])
    probtrue = probtrue/sum(probtrue)
    return probtrue

def herbasis(dim):
    pom1 = zeros([1,dim,dim])+1j*zeros([1,dim,dim])
    pom1[0] = identity(dim)
    arrays = [dot(transpose(pom1[0][[i]]),pom1[0][[i]]) for i in range(dim-1)]
    pom = stack(arrays,axis=0)
    her = concatenate((pom1,pom),axis=0)
    arrays = [dot(transpose(her[0][[i]]),her[0][[j]])+dot(transpose(her[0][[j]]),her[0][[i]]) for i in range(dim) for j in range(i+1,dim)]
    pom = stack(arrays,axis=0)
    her = concatenate((her,pom),axis=0)
    arrays = [-1j*dot(transpose(her[0][[i]]),her[0][[j]])+1j*dot(transpose(her[0][[j]]),her[0][[i]]) for i in range(dim) for j in range(i+1,dim)]
    pom = stack(arrays,axis=0)
    pom = concatenate((her,pom),axis=0)
    return pom

def gellmann(Q,dim):
    q = zeros([dim**2,dim,dim])+1j*zeros([dim**2,dim,dim])
    for i in range(dim**2):
        v = Q[i]
        for j in range(0,i):
            v = v-trace(v@q[j])*q[j]
        q[i] = v/sqrt(trace(v@v))
    return q

def conc(A):
    s = pauli()
    s2 = kron(s[2],s[2])
    At = (s2@conjugate(A))@s2
    As = sa.sqrtm(A)
    R = sa.sqrtm((As@At)@As)
    eigval = real(sort(la.eig(R)[0])[::-1])
    return max(0,eigval[0]-(eigval[1]+eigval[2]+eigval[3]))
    
def mubpom():
    p1 = array([1,0])
    p2 = array([0,1])
    mub = zeros([6,1,2])+1j*zeros([6,1,2])
    mub[0] = p1
    mub[1] = p2
    mub[2] = 1/sqrt(2)*(p1+p2)
    mub[3] = 1/sqrt(2)*(p1-p2)
    mub[4] = 1/sqrt(2)*(p1+1j*p2)
    mub[5] = 1/sqrt(2)*(p1-1j*p2)
    mubp = [transpose(mub[i])@conjugate(mub[i]) for i in range(6)]
    return mubp
    

def PauliForLearning(nProj):
	mubsN = linspace(0,35,nProj,dtype=int)
	mub = mubpom()
	mub2 = array([kron(mub[i],mub[j])/9 for i in range(6) for j in range(6)])
	mub3 = array([ mub2[n] for n in sort(mubsN[0:nProj]) ])
	return mub3

def dataGen(noStates,mub3,G,GAll):
	
	srmPOMs = array([ mub3 for k in range(noStates) ])
	
	radiusList=cbrt(cbrt(cbrt(random.rand(int(noStates/5))))) 
	rhon = randompure(4,int(noStates/5))
	stokesList = array([ radiusList[n]*stokesFromRho(rhon[n],G) for n in range(int(noStates/5)) ])
	rhoList1 = array([ rhoFromStokes(stokesList[n],G) for n in range(int(noStates/5)) ])
	
	rhoR1 = array([randomHaarState(4,1) for k in range(int(noStates/5))])
	rhoR2 = array([randomHaarState(4,2) for k in range(int(noStates/5))])
	rhoR3 = array([randomHaarState(4,3) for k in range(int(noStates/5))])
	rhoR4 = array([randomHaarState(4,4) for k in range(int(noStates/5))])
	rhoList = concatenate((rhoList1, rhoR1,rhoR2, rhoR3, rhoR4))
	
	indxs = random.shuffle(arange(noStates))
	rhoList = rhoList[indxs][0]
	
	# ~ concurrence
	y_train = array([conc(rhoList[n]) for n in range(int(4*noStates/5))])
	y_val = array([conc(rhoList[n]) for n in range(int(4*noStates/5),noStates)])
	
	# ~ probabilities
	probListSRM = array([ probdists(rhoList[n],srmPOMs[n]) for n in range(noStates)])

	
	x_train = array([probdists(rhoList[n],srmPOMs[n]) for n in range(int(4*noStates/5))])
	x_val = array([probdists(rhoList[n],srmPOMs[n]) for n in range(int(4*noStates/5),noStates)])
	
	return x_train,x_val,y_train,y_val

def myNet():
    model = Sequential()

    # ~ model.add(Conv1D(50,
                     # ~ kernel_size=17,
                     # ~ strides=17,
                     # ~ input_shape=(17*8,1),
                     # ~ activation='relu',
                     # ~ kernel_initializer=keras.initializers.glorot_normal(seed=42)
                     # ~ ))
    # ~ model.add(Flatten())
    
    model.add(Dense(100,
    input_shape=(nProj,),
    activation='relu',
    kernel_initializer=keras.initializers.glorot_normal(seed=42)
    ))
    
    model.add(Dense(120, activation='relu',kernel_initializer=keras.initializers.glorot_normal(seed=42)))
    model.add(Dense(80, activation='relu',kernel_initializer=keras.initializers.glorot_normal(seed=42)))
    model.add(Dense(60, activation='relu',kernel_initializer=keras.initializers.glorot_normal(seed=42)))
    model.add(Dense(60, activation='relu',kernel_initializer=keras.initializers.glorot_normal(seed=42)))
    model.add(Dense(60, activation='relu',kernel_initializer=keras.initializers.glorot_normal(seed=42)))
    model.add(Dense(40, activation='relu',kernel_initializer=keras.initializers.glorot_normal(seed=42)))
    
    model.add(Dense(1, activation='sigmoid',kernel_initializer=keras.initializers.glorot_normal(seed=42)))

    model.compile(loss='mse', optimizer="Nadam", metrics=['mean_absolute_error'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size, 
                        epochs=epochs,
                        verbose=0,
                        validation_data=(x_val, y_val)
                        )

    return(model,
           history.history['val_mean_absolute_error'][-1]
          )

nProjAll = 19
pom = array(range(1,nProjAll))[::-1]

dictModel={}
for k in pom:
    dictModel[str(k)]="ModelPauliConcProjR0v7"+str(2*k)
    
dictBestModel={}
for k in pom:
    dictBestModel[str(k)]="bestModelPauliConcProjR0v7"+str(2*k)

dim = 4
noStates = 600000
Q = herbasis(dim)
GAll = gellmann(Q,dim)*sqrt(dim)
G = GAll[1::]



for kk in pom:
	
	nProj = 2*kk
	print('nProj =', nProj)
	
	mub3 = PauliForLearning(nProj)
	savez("PauliR0"+str(nProj),mub3=mub3)
	
	x_train,x_val,y_train,y_val = dataGen(150000,mub3,G,GAll)
	
	batch_size = 500
	epochs = 50
	
	bestModel,bestMAE = myNet()
	bestModel.save(dictModel[str(kk)]+'.h5')
	
	for n in range(5):
	    print(n)
	    currentModel,currentMAE = myNet()
	    if currentMAE<bestMAE:
	        bestMAE=currentMAE
	        bestModel=currentModel
	        bestModel.save(dictModel[str(kk)]+'.h5')
	
	print(bestMAE)
			
	x_train,x_val,y_train,y_val = dataGen(noStates,mub3,G,GAll)
	
	filepath=dictBestModel[str(kk)]+'.h5'
	
	checkpoint = ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=0, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	
	history = bestModel.fit(x_train, y_train,
		batch_size=100,
		epochs=1000,
		verbose=0,
		validation_data=(x_val, y_val),
		callbacks = callbacks_list)
		
	print(history.history['val_mean_absolute_error'][-1])

	
	

