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
    
def getr():
    n = 0
    p = array(range(36)[::-1])**2
    p = p/sum(p)
    k = random.multinomial(1,p)
    while k[n]==0:
        n+=1
    return n,0
    
def myLog(M):
    U,S,VT = la.svd(M)
    vec = zeros(shape(M)[0])
    for n in range(shape(M)[0]):
        if S[n]>10**(-14):
            vec[n] = log2(S[n])
        else:
            vec[n] = 0
#     D = diag(log2(S))
    D = diag(vec)
    return (U@D)@VT

def mutInfM(rho):
    sA = -trace(parTrA(rho)@myLog(parTrA(rho)))
    sB= -trace(parTrB(rho)@myLog(parTrB(rho)))
    sAB= -trace(rho@(myLog(rho)))
    return real(1/2*(sA + sB - sAB))

def dataGen(noStates,G,GAll,ll):
	# ~ generate the training and the validation dataset
	mub = mubpom()
	mub2 = array([kron(mub[i],mub[j])/9 for i in range(6) for j in range(6) ])
	srmPOMs = array([ mub2 for k in range(noStates)])
	
	pomS = ones((noStates,36),dtype=int)
	
	for n in range(noStates):
	    myList=list(range(nProj))
	    random.shuffle(myList)
	    # ~ k = 0
	    k = getr()[ll]
	    for i in myList[0:k]:
	        srmPOMs[n][i] = zeros([4,4],complex)
	        pomS[n][i] = 0

	stokesFromPOMs = array([ array([ stokesFromRho(srmPOMs[n][m],GAll) for m in range(nProj) ]) for n in range(noStates) ])
	
	radiusList=cbrt(cbrt(cbrt(random.rand(int(noStates/5))))) # slightly deviated to pure states
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
	
	# ~ If you wish to train a network that predicts mutuan informaton, replace conc() with mutInfM()
	
	y_train = array([conc(rhoList[n]) for n in range(int(4*noStates/5))])
	y_val = array([conc(rhoList[n]) for n in range(int(4*noStates/5),noStates)])
	
	# ~ probabilities
	
	probListSRM = array([ probdists(rhoList[n],srmPOMs[n]) for n in range(noStates)])

	x_train = zeros([int(noStates*4/5),17*nProj,1])
	x_val = zeros([int(noStates/5),17*nProj,1])
	
	
	for n in range(int(noStates*4/5)):
		for i in range(nProj):
			x_train[n,i*17:i*17+16,0] = stokesFromPOMs[n][i]
			x_train[n,i*17+16,0] = probListSRM[n,i]
	# ~ print(x_train[0,:])
	
	# ~ stokesFromPOMs_val = stokesFromPOMs[int(noStates*4/5):noStates]
	# ~ proListSRM_val = probListSRM[int(noStates*4/5):noStates]
			
	for n in range(int(noStates/5)):
		for i in range(nProj):
			x_val[n,i*17:i*17+16,0] = stokesFromPOMs[int(noStates*4/5)+n][i]
			x_val[n,i*17+16,0] = probListSRM[int(noStates*4/5)+n,i]
	
	return x_train,x_val,y_train,y_val

def myNet():
	# ~ define a neural network model
    model = Sequential()

    model.add(Conv1D(100,
                     kernel_size=17,
                     strides=17,
                     input_shape=(17*36,1),
                     activation='relu',
                     kernel_initializer=keras.initializers.glorot_normal(seed=42)
                     ))
    model.add(Flatten())
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

    return(model,history.history['val_mean_absolute_error'][-1])

nProj = 36
dim = 4
noStates = 1000000
Q = herbasis(dim)
GAll = gellmann(Q,dim)*sqrt(dim)
G = GAll[1::]

x_train,x_val,y_train,y_val = dataGen(200000,G,GAll,1)


batch_size=500
epochs=100

# ~ pretraint the net for the full data 10 times, pick the best candidate
bestModel, bestMAE = myNet()
bestModel.save('bestModelConcHaar.h5')
print(bestMAE)
for n in range(10):
	currentModel, currentMAE = myNet()
	if currentMAE<bestMAE:
		bestMAE = currentMAE
		bestModel = currentModel
		print(bestMAE)
		bestModel.save('bestModelConcHaar.h5')

currentModel = load_model('bestModelConcHaar.h5')


##################################################################
############### standart learning #############################
################################################################

x_train,x_val,y_train,y_val = dataGen(noStates,G,GAll,0)

filepath='currentModelConcHaar.h5'

checkpoint = ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = currentModel.fit(x_train, y_train,
	batch_size=100,
	epochs=2000,
	verbose=1,
	validation_data=(x_val, y_val),
	callbacks = callbacks_list)

currentModel.save('bestModelConcHaar.h5')
mse = history.history['mean_absolute_error']
val_mse = history.history['val_mean_absolute_error']
savetxt('trainingErrorsConc.txt', [mse,val_mse], fmt="%s")

	
