import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scipy
import scipy.integrate as integrate
from scipy.integrate import quad
import scipy.special as special
from scipy.linalg import expm
from scipy.linalg import logm
from scipy.optimize import curve_fit
import seaborn as sns
import pandas as pd
#from lmfit import Model
from scipy.integrate import odeint
#from numpy import array, dot, pi
from scipy.integrate import complex_ode
from scipy import integrate
from tqdm import tqdm
from scipy.sparse.linalg import eigsh 
#from scipy.sparse import csc_matrix, kron, eye 

import time

# i-th operatior out of L qubits
def X(i,L):
    op=sp.csr_matrix([[0,1],[1,0]])
    if 1 < i  and i<L:
        Op_i=sp.kron(sp.kron(sp.eye(2**(i-1)),op),sp.eye(2**(L-i)))
    if i==1:
        Op_i=sp.kron(op,sp.eye(2**(L-1)))
    if i==L:
        Op_i=sp.kron(sp.eye(2**(L-1)),op)
        
    return Op_i

def Z(i,L):
    op=sp.csr_matrix([[1,0],[0,-1]])
    if 1 < i  and i<L:
        Zi=sp.kron(sp.kron(sp.eye(2**(i-1)),op),sp.eye(2**(L-i)))
    if i==1:
        Zi=sp.kron(op,sp.eye(2**(L-1)))
    if i==L:
        Zi=sp.kron(sp.eye(2**(L-1)),op)
    return Zi

def Y(i,L):
    op=sp.csr_matrix([[0,-1j],[1j,0]])
    if 1 < i  and i<L:
        Yi=sp.kron(sp.kron(sp.eye(2**(i-1)),op),sp.eye(2**(L-i)))
    if i==1:
        Yi=sp.kron(op,sp.eye(2**(L-1)))
    if i==L:
        Yi=sp.kron(sp.eye(2**(L-1)),op)
    return Yi


def V(Lam):
    #U1=sp.zeros([2*Lam,2*Lam])
    U1=sp.csr_matrix((2*Lam,2*Lam))
    U1[0,2*Lam-1]=1
    
    for i in range(2*Lam-1):
        U1[i+1,i]=1
        
    return U1


def V_dag(Lam):
    #U1=sp.zeros([2*Lam,2*Lam])
    U1=sp.csr_matrix((2*Lam,2*Lam))
    U1[2*Lam-1,0]=1
    
    for i in range(2*Lam-1):
        U1[i,i+1]=1
    
    return U1


# V is placed in the i-th position out of L
def U(i,Lam,L):
    op=V(Lam)
    op_i = None
    if 1 < i  and i<L:
        op_i=sp.kron(sp.kron(sp.eye((2*Lam)**(i-1)),op),sp.eye((2*Lam)**(L-i)))
    if i==1:
        op_i=sp.kron(op,sp.eye((2*Lam)**(L-1)))
    if i==L:
        op_i=sp.kron(sp.eye((2*Lam)**(L-1)),op)
        
    return op_i

def U_dag(i,Lam,L):
    op=V_dag(Lam)
    op_i = None
    if 1 < i  and i<L:
        op_i=sp.kron(sp.kron(sp.eye((2*Lam)**(i-1)),op),sp.eye((2*Lam)**(L-i)))
    if i==1:
        op_i=sp.kron(op,sp.eye((2*Lam)**(L-1)))
    if i==L:
        op_i=sp.kron(sp.eye((2*Lam)**(L-1)),op)
        
    return op_i



def E(Lam):
    U1=sp.csr_matrix((2*Lam,2*Lam))
    for i in range(2*Lam):
        U1[i,i]=-Lam+i
        
    return U1

# E^2(Lam) in the i-th out ouf L
def E2(i,Lam,L):
    X=E(Lam).dot(E(Lam))
    if 1 < i  and i<L:
        Xi=sp.kron(sp.kron(sp.eye((2*Lam)**(i-1)),X),sp.eye((2*Lam)**(L-i)))
    if i==1:
        Xi=sp.kron(X,sp.eye((2*Lam)**(L-1)))
    if i==L:
        Xi=sp.kron(sp.eye((2*Lam)**(L-1)),X)
        
    return Xi



def Ham_XX(Lam,L):
    H=sp.kron((U(L,Lam,L)+U_dag(L,Lam,L)), X(L,L).dot(X(1,L)))
    for i in range(1,L):
        H+=sp.kron((U(i,Lam,L)+U_dag(i,Lam,L)), X(i,L).dot(X(i+1,L)))
        
    return 0.25*H

def Ham_YY(Lam,L):
    H=sp.kron((U(L,Lam,L)+U_dag(L,Lam,L)), Y(L,L).dot(Y(1,L)))
    for i in range(1,L):
        H+=sp.kron((U(i,Lam,L)+U_dag(i,Lam,L)), Y(i,L).dot(Y(i+1,L)))
        
    return 0.25*H

def Ham_XY(Lam,L):
    H=sp.kron((U(L,Lam,L)-U_dag(L,Lam,L)), X(L,L).dot(Y(1,L)))
    for i in range(1,L):
        H+=sp.kron((U(i,Lam,L)-U_dag(i,Lam,L)), X(i,L).dot(Y(i+1,L)))
        
    return 0.25j*H

def Ham_YX(Lam,L):
    H=sp.kron((U(L,Lam,L)-U_dag(L,Lam,L)), Y(L,L).dot(X(1,L)))
    for i in range(1,L):
        H+=sp.kron((U(i,Lam,L)-U_dag(i,Lam,L)), Y(i,L).dot(X(i+1,L)))
        
    return -0.25j*H

def Ham_Z(m, Lam,L):
    H=sp.kron(sp.eye((2*Lam)**L) , Z(L,L))
    for i in range(1,L):
        H+=sp.kron(sp.eye((2*Lam)**L) , Z(i,L))
        
    return 0.5*m*H

def Ham_E(g, Lam,L):
    H=sp.kron( E2(L,Lam,L) , sp.eye(2**L))
    for i in range(1,L):
        H+=sp.kron( E2(i,Lam,L) , sp.eye(2**L))
        
    return (0.5*g**2)*H

def Ham_0(m,g, Lam,L):
    #return Ham_XX(Lam,L)+Ham_XY(Lam,L)+Ham_YX(Lam,L)+Ham_XX(Lam,L)+Ham_Z(m,Lam,L)+Ham_E(g,Lam,L)
    return Ham_XX(Lam,L)+Ham_XY(Lam,L)+Ham_YX(Lam,L)+Ham_YY(Lam,L)+Ham_Z(m,Lam,L)+Ham_E(g,Lam,L)


#from scipy import sparse
#from scipy.sparse import csr_matrix
from scipy import sparse
from scipy.sparse import csr_matrix
def ext(t,Lam,L,k):
    OP=sp.eye((2*Lam)**L)

    #if t==0:
        #OP=U(1,Lam,L)
    #OP=sparse.csr_matrix(np.eye((2*Lam)**L))
    
    if t==1:
        OP = U(2,Lam,L).dot( U(L,Lam,L) )
        
    elif t==2:
        OP = (U(3,Lam,L)**2).dot(U(L-1,Lam,L)**2) 
        
    elif t==3:
        OP =  (U(4,Lam,L)**3).dot(U(L-2,Lam,L)**3)    
        
    elif t==4:
        OP = (U(5,Lam,L)**4).dot(U(L-3,Lam,L)**4)

    elif t==5:
        OP = (U(6,Lam,L)**5).dot(U(L-4,Lam,L)**5) 
        
    elif t==6:
        OP = (U(7,Lam,L)**6).dot(U(L-5,Lam,L)**6)
        
    elif t==7:
        OP = (U(8,Lam,L)**7).dot(U(L-6,Lam,L)**7)
        
    elif t==8:
        OP = (U(9,Lam,L)**8).dot(U(L-7,Lam,L)**8)
    
    else:
        OP=sp.eye((2*Lam)**L)
        
        
    
    #OP=sp.kron(OP,sp.eye(2**L))
    #OP = OP.toarray()
    
    #for i in range(k):
    #    OP=scipy.linalg.sqrtm(OP)
    
    #return OP



    
    OP = OP.toarray()
    
    for i in range(k):
        OP=scipy.linalg.sqrtm(OP)
    
    OP = sp.csr_matrix(OP)
    OP=sp.kron(OP,sp.eye(2**L))
    
    return OP



def evolv_XX(t,Lam,L,k):
    return sp.linalg.expm(-1j*(t/k)*Ham_XX(Lam,L))

def evolv_XY(t,Lam,L,k):
    return sp.linalg.expm(-1j*(t/k)*Ham_XY(Lam,L))

def evolv_YX(t,Lam,L,k):
    return sp.linalg.expm(-1j*(t/k)*Ham_YX(Lam,L))

def evolv_YY(t,Lam,L,k):
    return sp.linalg.expm(-1j*(t/k)*Ham_YY(Lam,L))

def evolv_Z(t,m, Lam,L,k):
    return sp.linalg.expm(-1j*(t/k)*Ham_Z(m,Lam,L))

def evolv_E(t,g, Lam,L,k):
    return sp.linalg.expm(-1j*(t/k)*Ham_E(g, Lam,L))


def evolv(t,m,g,Lam,L,k):
    M=evolv_XX(t,Lam,L,k).dot(evolv_XY(t,Lam,L,k))
    M=M.dot(evolv_YX(t,Lam,L,k))
    M=M.dot(evolv_YY(t,Lam,L,k))
    M=M.dot(evolv_Z(t,m,Lam,L,k))
    M=M.dot(evolv_E(t,g,Lam,L,k))
    M=M.dot(ext(t,Lam,L,k))
    
    return M**k
    

def Ene(t,m,g,Lam,L,k):
    eig_val, eig_vec = sp.linalg.eigsh(Ham_0(m,g, Lam,L), k=1, which="SA")
    idx = eig_val.argsort()[::1]
    eig_val_sorted = eig_val[idx]
    eig_vec_sorted =np.transpose(eig_vec[:,idx])
    
    u=eig_vec_sorted[0]#/float(np.sqrt(np.dot(np.conjugate(eig_vec_sorted[0]),eig_vec_sorted[0])))
    
    #return np.dot(np.conjugate(eig_vec_sorted[0]),np.dot(commn, eig_vec_sorted[0]))
    
    
    #return np.dot(np.transpose(np.conjugate(u)),np.dot(evolv(t,m,g,Lam,L,k), u))
    return np.dot(np.transpose(np.conjugate(u)), evolv(t,m,g,Lam,L,k).dot(u))
    
    
    
    
    
    
    
    
    
    
###########################################################################################    
#
# New implmentations
#
###########################################################################################
    
def expM_comp(t,k,A,v,d=10):
    
    s = v.astype(complex)
    u = v
    
    for i in range(d):
        u = A.dot(u) # (A**d)(v)
        s += u / float(np.math.factorial(i+1))
        
    return s
    
    
    
def Ene_new2(t,m,g,Lam,L,k, ext_offT):
    eig_val, eig_vec = sp.linalg.eigsh(Ham_0(m,g, Lam,L), k=1, which="SA")
    idx = eig_val.argsort()[::1]
    eig_val_sorted = eig_val[idx]
    eig_vec_sorted =np.transpose(eig_vec[:,idx])
    
    u=eig_vec_sorted[0]#/float(np.sqrt(np.dot(np.conjugate(eig_vec_sorted[0]),eig_vec_sorted[0])))
    
    #return np.dot(np.conjugate(eig_vec_sorted[0]),np.dot(commn, eig_vec_sorted[0]))

    
    M_ext = None
        
    if t==1 or t==2 or t==3 or t==4 or t==5 or t==6 or t==7 or t==8 or t==9 or t==10:
        M_ext = ext(t,Lam,L,k)
    else:
        M_ext = ext_offT
    
    XX_exp = evolv_XX(t,Lam,L,k)
    XY_exp = evolv_XY(t,Lam,L,k)
    YX_exp = evolv_YX(t,Lam,L,k)
    YY_exp = evolv_YY(t,Lam,L,k)
    Z_exp = evolv_Z(t,m,Lam,L,k)
    E_exp = evolv_E(t,g,Lam,L,k)
    
    
    v = u
    
    for i in range(k):
        v = M_ext.dot(v)
        v = E_exp.dot(v)
        v = Z_exp.dot(v)
        v = YY_exp.dot(v)
        v = YX_exp.dot(v)
        v = XY_exp.dot(v)
        v = XX_exp.dot(v)
    
    return np.dot(np.transpose(np.conjugate(u)), v)
    

    
def Ene_new3(t,m,g,Lam,L,k):
    eig_val, eig_vec = sp.linalg.eigsh(Ham_0(m,g, Lam,L), k=1, which="SA")
    idx = eig_val.argsort()[::1]
    eig_val_sorted = eig_val[idx]
    eig_vec_sorted =np.transpose(eig_vec[:,idx])
    
    u=eig_vec_sorted[0]#/float(np.sqrt(np.dot(np.conjugate(eig_vec_sorted[0]),eig_vec_sorted[0])))
    
    #return np.dot(np.conjugate(eig_vec_sorted[0]),np.dot(commn, eig_vec_sorted[0]))

    
    XX_exp = evolv_XX(t,Lam,L,k)
    XY_exp = evolv_XY(t,Lam,L,k)
    YX_exp = evolv_YX(t,Lam,L,k)
    YY_exp = evolv_YY(t,Lam,L,k)
    Z_exp = evolv_Z(t,m,Lam,L,k)
    E_exp = evolv_E(t,g,Lam,L,k)
    
    v = u
        
    if t==1 or t==2 or t==3 or t==4 or t==5 or t==6 or t==7 or t==8 or t==9 or t==10:
        M_ext = ext(t,Lam,L,k)
        
        for i in range(k):
            v = M_ext.dot(v)
            v = E_exp.dot(v)
            v = Z_exp.dot(v)
            v = YY_exp.dot(v)
            v = YX_exp.dot(v)
            v = XY_exp.dot(v)
            v = XX_exp.dot(v)
        
        
    else:

        for i in range(k):
            v = E_exp.dot(v)
            v = Z_exp.dot(v)
            v = YY_exp.dot(v)
            v = YX_exp.dot(v)
            v = XY_exp.dot(v)
            v = XX_exp.dot(v)

    
    
   
    return np.dot(np.transpose(np.conjugate(u)), v)
    


    
def Ene_new4(t,m,g,Lam,L,k):
    eig_val, eig_vec = sp.linalg.eigsh(Ham_0(m,g, Lam,L), k=1, which="SA")
    idx = eig_val.argsort()[::1]
    eig_val_sorted = eig_val[idx]
    eig_vec_sorted =np.transpose(eig_vec[:,idx])
    
    u=eig_vec_sorted[0]#/float(np.sqrt(np.dot(np.conjugate(eig_vec_sorted[0]),eig_vec_sorted[0])))
    
    #return np.dot(np.conjugate(eig_vec_sorted[0]),np.dot(commn, eig_vec_sorted[0]))

    
    v = u
        
    if t==1 or t==2 or t==3 or t==4 or t==5 or t==6 or t==7 or t==8 or t==9 or t==10:
        M_ext = ext(t,Lam,L,k)
        
        for i in range(k):
            v = M_ext.dot(v)
            v = expM_comp(t,k,(-1j*(t/k))*Ham_E(g, Lam,L),v)
            v = expM_comp(t,k,(-1j*(t/k))*Ham_Z(m, Lam,L),v)
            v = expM_comp(t,k,(-1j*(t/k))*Ham_YY(Lam,L),v)
            v = expM_comp(t,k,(-1j*(t/k))*Ham_YX(Lam,L),v)
            v = expM_comp(t,k,(-1j*(t/k))*Ham_XY(Lam,L),v)
            v = expM_comp(t,k,(-1j*(t/k))*Ham_XX(Lam,L),v)
        
        
    else:

        for i in range(k):
            v = expM_comp(t,k,(-1j*(t/k))*Ham_E(g, Lam,L),v)
            v = expM_comp(t,k,(-1j*(t/k))*Ham_Z(m, Lam,L),v)
            v = expM_comp(t,k,(-1j*(t/k))*Ham_YY(Lam,L),v)
            v = expM_comp(t,k,(-1j*(t/k))*Ham_YX(Lam,L),v)
            v = expM_comp(t,k,(-1j*(t/k))*Ham_XY(Lam,L),v)
            v = expM_comp(t,k,(-1j*(t/k))*Ham_XX(Lam,L),v)

    
   
    return np.dot(np.transpose(np.conjugate(u)), v)
    

    
    
    

def evolv_new(t,m,g,Lam,L):
    return expm(-1j*t*Ham_0(m,Lam,L))

def evolv_A(t,A):
    return expm(-1j*t*A)



def Ene_new(t,A, u):
    #eig_val, eig_vec=eigsh(A, k=1, which="SA")
    #idx = eig_val.argsort()[::1]
    #eig_val_sorted = eig_val[idx]
    #eig_vec_sorted =np.transpose(eig_vec[:,idx])
    
    #u=eig_vec_sorted[0]#/float(np.sqrt(np.dot(np.conjugate(eig_vec_sorted[0]),eig_vec_sorted[0])))
    
    #return np.dot(np.conjugate(eig_vec_sorted[0]),np.dot(commn, eig_vec_sorted[0]))
    
    return np.dot(np.conjugate(u), np.dot(evolv_A(t,A), u))
    






