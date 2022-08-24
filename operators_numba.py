import numpy as np
from numba import jit
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
from numpy import array, dot, pi
from scipy.integrate import complex_ode
from scipy import integrate
from tqdm import tqdm
from scipy.sparse.linalg import eigsh 

import time

# i-th operatior out of L qubits
@jit
def X(i,L):
    op=np.array([[0,1],[1,0]])
    if 1 < i  and i<L:
        Op_i=np.kron(np.kron(np.eye(2**(i-1)),op),np.eye(2**(L-i)))
    if i==1:
        Op_i=np.kron(op,np.eye(2**(L-1)))
    if i==L:
        Op_i=np.kron(np.eye(2**(L-1)),op)
        
    return Op_i

@jit
def Z(i,L):
    op=np.array([[1,0],[0,-1]])
    if 1 < i  and i<L:
        Zi=np.kron(np.kron(np.eye(2**(i-1)),op),np.eye(2**(L-i)))
    if i==1:
        Zi=np.kron(op,np.eye(2**(L-1)))
    if i==L:
        Zi=np.kron(np.eye(2**(L-1)),op)
    return Zi

@jit
def Y(i,L):
    op=np.array([[0,-1j],[1j,0]])
    if 1 < i  and i<L:
        Yi=np.kron(np.kron(np.eye(2**(i-1)),op),np.eye(2**(L-i)))
    if i==1:
        Yi=np.kron(op,np.eye(2**(L-1)))
    if i==L:
        Yi=np.kron(np.eye(2**(L-1)),op)
    return Yi


@jit
def V(Lam):
    U1=np.zeros([2*Lam,2*Lam])
    U1[0,2*Lam-1]=1
    
    for i in range(2*Lam-1):
        U1[i+1,i]=1
        
    return U1


@jit
def V_dag(Lam):
    U1=np.zeros([2*Lam,2*Lam])
    U1[2*Lam-1,0]=1
    
    for i in range(2*Lam-1):
        U1[i,i+1]=1
    
    return U1


# V is placed in the i-th position out of L
@jit
def U(i,Lam,L):
    op=V(Lam)
    if 1 < i  and i<L:
        op_i=np.kron(np.kron(np.eye((2*Lam)**(i-1)),op),np.eye((2*Lam)**(L-i)))
    if i==1:
        op_i=np.kron(op,np.eye((2*Lam)**(L-1)))
    if i==L:
        op_i=np.kron(np.eye((2*Lam)**(L-1)),op)
        
    return op_i

@jit
def U_dag(i,Lam,L):
    op=V_dag(Lam)
    if 1 < i  and i<L:
        op_i=np.kron(np.kron(np.eye((2*Lam)**(i-1)),op),np.eye((2*Lam)**(L-i)))
    if i==1:
        op_i=np.kron(op,np.eye((2*Lam)**(L-1)))
    if i==L:
        op_i=np.kron(np.eye((2*Lam)**(L-1)),op)
        
    return op_i


@jit
def E(Lam):
    U1=np.zeros([2*Lam,2*Lam])
    for i in range(2*Lam):
        U1[i,i]=-Lam+i
        
    return U1

# E^2(Lam) in the i-th out ouf L
@jit
def E2(i,Lam,L):
    X=E(Lam).dot(E(Lam)) # E^2(Lam)
    if 1 < i  and i<L:
        Xi=np.kron(np.kron(np.eye((2*Lam)**(i-1)),X),np.eye((2*Lam)**(L-i)))
    if i==1:
        Xi=np.kron(X,np.eye((2*Lam)**(L-1)))
    if i==L:
        Xi=np.kron(np.eye((2*Lam)**(L-1)),X)
        
    return Xi



@jit
def Ham_XX(Lam,L):
    H=np.kron((U(L,Lam,L)+U_dag(L,Lam,L)), X(L,L).dot(X(1,L)))
    for i in range(1,L):
        H+=np.kron((U(i,Lam,L)+U_dag(i,Lam,L)), X(i,L).dot(X(i+1,L)))
        
    return 0.25*H

@jit
def Ham_YY(Lam,L):
    H=np.kron((U(L,Lam,L)+U_dag(L,Lam,L)), Y(L,L).dot(Y(1,L)))
    for i in range(1,L):
        H+=np.kron((U(i,Lam,L)+U_dag(i,Lam,L)), Y(i,L).dot(Y(i+1,L)))
        
    return 0.25*H

@jit
def Ham_XY(Lam,L):
    H=np.kron((U(L,Lam,L)-U_dag(L,Lam,L)), X(L,L).dot(Y(1,L)))
    for i in range(1,L):
        H+=np.kron((U(i,Lam,L)-U_dag(i,Lam,L)), X(i,L).dot(Y(i+1,L)))
        
    return 0.25j*H

@jit
def Ham_YX(Lam,L):
    H=np.kron((U(L,Lam,L)-U_dag(L,Lam,L)), Y(L,L).dot(X(1,L)))
    for i in range(1,L):
        H+=np.kron((U(i,Lam,L)-U_dag(i,Lam,L)), Y(i,L).dot(X(i+1,L)))
        
    return -0.25j*H

@jit
def Ham_Z(m, Lam,L):
    H=np.kron(np.eye((2*Lam)**L) , Z(L,L))
    for i in range(1,L):
        H+=np.kron(np.eye((2*Lam)**L) , Z(i,L))
        
    return 0.5*m*H

@jit
def Ham_E(g, Lam,L):
    H=np.kron( E2(L,Lam,L) , np.eye(2**L))
    for i in range(1,L):
        H+=np.kron( E2(i,Lam,L) , np.eye(2**L))
        
    return (0.5*g**2)*H

@jit
def Ham_0(m,g, Lam,L):
    #return Ham_XX(Lam,L)+Ham_XY(Lam,L)+Ham_YX(Lam,L)+Ham_XX(Lam,L)+Ham_Z(m,Lam,L)+Ham_E(g,Lam,L)
    return Ham_XX(Lam,L)+Ham_XY(Lam,L)+Ham_YX(Lam,L)+Ham_YY(Lam,L)+Ham_Z(m,Lam,L)+Ham_E(g,Lam,L)


#from scipy import sparse
#from scipy.sparse import csr_matrix
@jit
def ext(t,Lam,L,k):
    OP=np.eye((2*Lam)**L)
    #OP=sparse.csr_matrix(np.eye((2*Lam)**L))
    
    if t> 1:
        OP=OP.dot(dot(U(1,Lam,L), U(L,Lam,L) ))
        
    if t>2:
        OP=OP.dot(dot(np.linalg.matrix_power(U(2,Lam,L), 2), np.linalg.matrix_power(U(L-1,Lam,L), 2)))
        
    if t>3:
        OP=OP.dot(dot(np.linalg.matrix_power(U(3,Lam,L), 3), np.linalg.matrix_power(U(L-2,Lam,L), 3)))
        
    if t>4:
        OP=OP.dot(dot(np.linalg.matrix_power(U(4,Lam,L), 4), np.linalg.matrix_power(U(L-3,Lam,L), 4)))

    if t>5:
        OP=OP.dot(dot(np.linalg.matrix_power(U(5,Lam,L), 5), np.linalg.matrix_power(U(L-4,Lam,L), 5)))
        
    if t>6:
        OP=OP.dot(dot(np.linalg.matrix_power(U(6,Lam,L), 6), np.linalg.matrix_power(U(L-5,Lam,L), 6)))
            
    OP=np.kron(OP,np.eye(2**L))
    
    #start_time = time.time()
    for i in range(k):
        OP=scipy.linalg.sqrtm(OP)
    #print("--- Matrxi Sqrt time: %s seconds ---" % (time.time() - start_time))
    
    return OP



@jit
def evolv_XX(t,Lam,L,k):
    return expm(-1j*(t/k)*Ham_XX(Lam,L))

@jit
def evolv_XY(t,Lam,L,k):
    return expm(-1j*(t/k)*Ham_XY(Lam,L))

@jit
def evolv_YX(t,Lam,L,k):
    return expm(-1j*(t/k)*Ham_YX(Lam,L))

@jit
def evolv_YY(t,Lam,L,k):
    return expm(-1j*(t/k)*Ham_YY(Lam,L))

@jit
def evolv_Z(t,m, Lam,L,k):
    return expm(-1j*(t/k)*Ham_Z(m,Lam,L))

@jit
def evolv_E(t,g, Lam,L,k):
    return expm(-1j*(t/k)*Ham_E(g, Lam,L))


@jit
def evolv(t,m,g,Lam,L,k):
    M=evolv_XX(t,Lam,L,k).dot(evolv_XY(t,Lam,L,k))
    M=M.dot(evolv_YX(t,Lam,L,k))
    M=M.dot(evolv_YY(t,Lam,L,k))
    M=M.dot(evolv_Z(t,m,Lam,L,k))
    M=M.dot(evolv_E(t,g,Lam,L,k))
    M=M.dot(ext(t,Lam,L,k))
    
    return np.linalg.matrix_power(M, k)
    

@jit
def Ene(t,m,g,Lam,L,k):
    eig_val, eig_vec=eigsh(Ham_0(m,g, Lam,L), k=1, which="SA")
    idx = eig_val.argsort()[::1]
    eig_val_sorted = eig_val[idx]
    eig_vec_sorted =np.transpose(eig_vec[:,idx])
    
    u=eig_vec_sorted[0]#/float(np.sqrt(np.dot(np.conjugate(eig_vec_sorted[0]),eig_vec_sorted[0])))
    
    #return np.dot(np.conjugate(eig_vec_sorted[0]),np.dot(commn, eig_vec_sorted[0]))
    
    
    return np.dot(np.conjugate(u),np.dot(evolv(t,m,g,Lam,L,k), u))
    
    
    
    
    
    
    
    
    
    
###########################################################################################    
#
# New implmentations
#
###########################################################################################
    
    
    
    
    
    



@jit
def ext_new(t,Lam,L,k, U1, U2, U3, U4, U5):
    OP=np.eye((2*Lam)**L)
    #OP=sparse.csr_matrix(np.eye((2*Lam)**L))
    
    if t> 1:
        OP=OP.dot(U1)
        
    if t>2:
        OP=OP.dot(U2)
        
    if t>3:
        OP=OP.dot(U3)
        
    if t>4:
        OP=OP.dot(U4)

    if t>5:
        OP=OP.dot(U5)
        
    if t>6:
        OP=OP.dot(U6)
            
    OP=np.kron(OP,np.eye(2**L))
    
    #start_time = time.time()
    for i in range(k):
        OP=scipy.linalg.sqrtm(OP)
    #print("--- Matrxi Sqrt time: %s seconds ---" % (time.time() - start_time))

    
    return OP


@jit
def evolv_new(t,m,g,Lam,L):
    return expm(-1j*t*Ham_0(m,Lam,L))

@jit
def evolv_A(t,A):
    return expm(-1j*t*A)



@jit
def Ene_new(t,A, u):
    #eig_val, eig_vec=eigsh(A, k=1, which="SA")
    #idx = eig_val.argsort()[::1]
    #eig_val_sorted = eig_val[idx]
    #eig_vec_sorted =np.transpose(eig_vec[:,idx])
    
    #u=eig_vec_sorted[0]#/float(np.sqrt(np.dot(np.conjugate(eig_vec_sorted[0]),eig_vec_sorted[0])))
    
    #return np.dot(np.conjugate(eig_vec_sorted[0]),np.dot(commn, eig_vec_sorted[0]))
    
    return np.dot(np.conjugate(u), np.dot(evolv_A(t,A), u))
    






