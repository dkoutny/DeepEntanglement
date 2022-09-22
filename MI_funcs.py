# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:04:47 2022

@author: Dominik
"""


from numpy import *
from numpy import linalg as la
from scipy import linalg as sa

def myLog(M):
    U,S,VT = la.svd(M)
    for n in range(shape(S)[0]):
        if S[n]<10**(-14):
            S[n] = 1
    D = diag(log2(S))
    return (U@D)@VT

def parTrBCD4q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(id2,b[i]),b[j]),b[k]) for i in range(2) for j in range(2) for k in range(2) ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(8)]),axis=0)
    return rhoB

def parTrACD4q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(b[i],id2),b[j]),b[k]) for i in range(2) for j in range(2) for k in range(2) ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(8)]),axis=0)
    return rhoB

def parTrABD4q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(b[i],b[j]),id2),b[k]) for i in range(2) for j in range(2) for k in range(2)])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(8)]),axis=0)
    return rhoB

def parTrABC4q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(b[i],b[j]),b[k]),id2) for i in range(2) for j in range(2) for k in range(2) ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(8)]),axis=0)
    return rhoB

def parTrAB4q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(b[i],b[j]),id2),id2) for i in range(2) for j in range(2) ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(4)]),axis=0)
    return rhoB

def parTrAC4q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(b[i],id2),b[j]),id2) for i in range(2) for j in range(2) ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(4)]),axis=0)
    return rhoB

def parTrAD4q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(b[i],id2),id2),b[j]) for i in range(2) for j in range(2) ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(4)]),axis=0)
    return rhoB

def parTrBC4q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(id2,b[i]),b[j]),id2) for i in range(2) for j in range(2) ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(4)]),axis=0)
    return rhoB

def parTrBD4q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(id2,b[i]),id2),b[j]) for i in range(2) for j in range(2) ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(4)]),axis=0)
    return rhoB

def parTrCD4q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(id2,id2),b[i]),b[j]) for i in range(2) for j in range(2) ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(4)]),axis=0)
    return rhoB

def mutInf4q(rho):
    sA = -trace(parTrBCD4q(rho)@myLog(parTrBCD4q(rho)))
    sB= -trace(parTrACD4q(rho)@myLog(parTrACD4q(rho)))
    sC= -trace(parTrABD4q(rho)@myLog(parTrABD4q(rho)))
    sD= -trace(parTrABC4q(rho)@myLog(parTrABC4q(rho)))
    sAB= -trace(parTrCD4q(rho)@myLog(parTrCD4q(rho)))
    sAC= -trace(parTrBD4q(rho)@myLog(parTrBD4q(rho)))
    sAD= -trace(parTrBC4q(rho)@myLog(parTrBC4q(rho)))
    sBC= -trace(parTrAD4q(rho)@myLog(parTrAD4q(rho)))
    sBD= -trace(parTrAC4q(rho)@myLog(parTrAC4q(rho)))
    sCD= -trace(parTrAD4q(rho)@myLog(parTrAD4q(rho)))
    return array([real(1/2*(sA + sB - sAB)),real(1/2*(sA + sC - sAC)),real(1/2*(sA + sD - sAD)),
                 real(1/2*(sB + sC - sBC)),real(1/2*(sB + sD - sBD)),real(1/2*(sC + sD - sCD))])

def parTrBCDE5q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(kron(id2,b[i]),b[j]),b[k]),b[l]) for i in range(2) for j in range(2) for k in range(2) for l in range(2) ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(16)]),axis=0)
    return rhoB

def parTrACDE5q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(kron(b[i],id2),b[j]),b[k]),b[l]) for i in range(2) for j in range(2) for k in range(2) for l in range(2) ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(16)]),axis=0)
    return rhoB

def parTrABDE5q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(kron(b[i],b[j]),id2),b[k]),b[l]) for i in range(2) for j in range(2) for k in range(2) for l in range(2) ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(16)]),axis=0)
    return rhoB

def parTrABCE5q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(kron(b[i],b[j]),b[k]),id2),b[l]) for i in range(2) for j in range(2) for k in range(2)  for l in range(2) ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(16)]),axis=0)
    return rhoB
               
def parTrABCD5q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(kron(b[i],b[j]),b[k]),b[l]),id2) for i in range(2) for j in range(2) for k in range(2)  for l in range(2) ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(16)]),axis=0)
    return rhoB

def parTrABC5q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(kron(b[i],b[j]),b[k]),id2),id2) for i in range(2) for j in range(2)  for k in range(2)  ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(8)]),axis=0)
    return rhoB

def parTrABD5q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(kron(b[i],b[j]),id2),b[k]),id2) for i in range(2) for j in range(2) for k in range(2)  ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(8)]),axis=0)
    return rhoB

def parTrABE5q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(kron(b[i],b[j]),id2),b[k]),id2) for i in range(2) for j in range(2)  for k in range(2) ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(8)]),axis=0)
    return rhoB


def parTrACD5q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(kron(b[i],id2),b[j]),b[k]),id2) for i in range(2) for j in range(2) for k in range(2) ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(8)]),axis=0)
    return rhoB

def parTrACE5q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(kron(b[i],id2),b[j]),id2),b[k]) for i in range(2) for j in range(2)  for k in range(2) ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(8)]),axis=0)
    return rhoB


def parTrADE5q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(kron(b[i],id2),id2),b[j]),b[k]) for i in range(2) for j in range(2) for k in range(2)  ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(8)]),axis=0)
    return rhoB

def parTrBCD5q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(kron(id2,b[i]),b[j]),b[k]),id2) for i in range(2) for j in range(2) for k in range(2) ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(8)]),axis=0)
    return rhoB

def parTrBCE5q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(kron(id2,b[i]),b[j]),id2),b[k]) for i in range(2) for j in range(2) for k in range(2) ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(8)]),axis=0)
    return rhoB

def parTrBDE5q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(kron(id2,b[i]),id2),b[j]),b[k]) for i in range(2) for j in range(2) for k in range(2) ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(8)]),axis=0)
    return rhoB

def parTrCDE5q(rho):
    b=zeros([2,2,1])
    b[0]=array([[1],[0]])
    b[1]=array([[0],[1]])
    id2 = identity(2)
    A = array([ kron(kron(kron(kron(id2,id2),b[i]),b[j]),b[k]) for i in range(2) for j in range(2) for k in range(2) ])
    rhoB = sum(array([A[i].transpose()@rho@A[i] for i in range(8)]),axis=0)
    return rhoB

def mutInf5q(rho):
    sA = -trace(parTrBCDE5q(rho)@myLog(parTrBCDE5q(rho)))
    sB= -trace(parTrACDE5q(rho)@myLog(parTrACDE5q(rho)))
    sC= -trace(parTrABDE5q(rho)@myLog(parTrABDE5q(rho)))
    sD= -trace(parTrABCE5q(rho)@myLog(parTrABCE5q(rho)))
    sE= -trace(parTrABCD5q(rho)@myLog(parTrABCD5q(rho)))
    sAB= -trace(parTrCDE5q(rho)@myLog(parTrCDE5q(rho)))
    sAC= -trace(parTrBDE5q(rho)@myLog(parTrBDE5q(rho)))
    sAD= -trace(parTrBCE5q(rho)@myLog(parTrBCE5q(rho)))
    sAE= -trace(parTrBCD5q(rho)@myLog(parTrBCD5q(rho)))
    sBC= -trace(parTrADE5q(rho)@myLog(parTrADE5q(rho)))
    sBD= -trace(parTrACE5q(rho)@myLog(parTrACE5q(rho)))
    sBE= -trace(parTrACD5q(rho)@myLog(parTrACD5q(rho)))
    sCD= -trace(parTrABE5q(rho)@myLog(parTrABE5q(rho)))
    sCE= -trace(parTrABD5q(rho)@myLog(parTrABD5q(rho)))
    sDE= -trace(parTrABC5q(rho)@myLog(parTrABC5q(rho)))
    return array([real(1/2*(sA + sB - sAB)),real(1/2*(sA + sC - sAC)),real(1/2*(sA + sD - sAD)),
                  real(1/2*(sA + sE - sAE)),
                 real(1/2*(sB + sC - sBC)),real(1/2*(sB + sD - sBD)),real(1/2*(sB + sE - sBE)),
                  real(1/2*(sC + sD - sCD)),real(1/2*(sC + sE - sCE)),real(1/2*(sD + sE - sDE))])