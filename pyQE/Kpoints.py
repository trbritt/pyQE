# -*- coding: utf-8 -*-
#=========================================================
# Beginning of Kpoints.py
# @author: Tristan Britt
# @email: tristan.britt@mail.mcgill.ca
# @description: This file contains all functions to generate
# reciprocal space data for a given crystal symmetry
#
# This software is part of a package distributed under the 
# GPLv3 license, see LICENSE.txt
#=========================================================

import numpy as np
###Kpoint paths taken from "W. Setyawan, S. Curtarolo, arXiv:1004.2974v1 [cond-mat.mtrl-sci] 17 April 2010.""

###Get the length of a vector between two points in three-dimensions
def Length(Point1,Point2):
    return np.sqrt(np.sum((Point2-Point1)**2))

###Reorder the parameters according to the paper (see above)
def order_parameters(celldm_1,celldm_2,celldm_3):
    if celldm_2<1:
        if celldm_3 < 1 and celldm_3 < celldm_2:
            a=celldm_1
            b=celldm_2*celldm_1
            c=celldm_3*celldm_1
        elif celldm_3 < 1 and celldm_3 > celldm_2:
            a=celldm_1
            b=celldm_3*celldm_1
            c=celldm_2*celldm_1
        else:
            a=celldm_3*celldm_1
            b=celldm_1
            c=celldm_2*celldm_3
    else:
        if celldm_3 > celldm_2:
            a=celldm_3*celldm_1
            b=celldm_2*celldm_1
            c=celldm_1
        elif celldm_3 >1:
            a=celldm_2*celldm_1
            b=celldm_3*celldm_1
            c=celldm_1
        else:
            a=celldm_2*celldm_1
            b=celldm_1
            c=celldm_3*celldm_1
    return a,b,c

###Each Bravais lattice has its own k-point path, factor is the minimum number between two points
def Kpoint_path(ibrav,factor,celldm_1=10,celldm_2=1,celldm_3=1,alpha=90,beta=90,gamma=90):
    if ibrav == 1:
        Kpoint_path,data=Kpoint_cubic(factor)
    elif ibrav == 2:
        Kpoint_path,data=Kpoint_fcc(factor)
    elif ibrav == 3:
        Kpoint_path,data=Kpoint_bcc(factor)
    elif ibrav == 4:
        Kpoint_path,data=Kpoint_hexagonal(factor,celldm_3)
    elif ibrav == abs(5):
        Kpoint_path,data=Kpoint_rhombohedral(factor,gamma)
    elif ibrav == 6:
        Kpoint_path,data=Kpoint_tetragonal(factor)
    elif ibrav == 7:
        Kpoint_path,data=Kpoint_body_tetragonal(factor,celldm_3)
    elif ibrav == 8:
        Kpoint_path,data=Kpoint_orthorhombic(factor)
    elif ibrav == abs(9):
        Kpoint_path,data=Kpoint_c_orthorhombic(factor,celldm_1,celldm_2,celldm_3)
    elif ibrav == 10:
        Kpoint_path,data=Kpoint_face_orthorhombic(factor,celldm_1,celldm_2,celldm_3)
    elif ibrav == 11:
        Kpoint_path,data=Kpoint_body_orthorhombic(factor,celldm_1,celldm_2,celldm_3)
    elif ibrav == abs(12):
        Kpoint_path,data=Kpoint_monoclinic(factor,celldm_1,celldm_2,celldm_3,gamma)
    elif ibrav == abs(13):
        Kpoint_path,data=Kpoint_c_monoclinic(factor,celldm_1,celldm_2,celldm_3,gamma)
    elif ibrav == 14:
        Kpoint_path,data=Kpoint_triclinic(factor,celldm_1,celldm_2,celldm_3,alpha,beta,gamma)
    else:
        Kpoint_path,data=[],[]
        print("ibrav does not exist!")
    return Kpoint_path,data 

###k-point path for primitive cubic
def Kpoint_cubic(factor):
    #Gamma-X-M-Gamma-R-X|M-R
    List=['$\Gamma$',"$X$","$M$","$\Gamma$",'$R$',"$X$|$M$","$R$"]
    distance=[]
    Gamma=[0,0,0]
    M=[0.5,0.5,0]
    X=[0,0.5,0]
    R=[0.5,0.5,0.5]
    distance.append(Length(Gamma,X))
    distance.append(Length(X,M))
    distance.append(Length(M,Gamma))
    distance.append(Length(Gamma,R))
    distance.append(Length(R,X))
    distance.append(Length(M,R))
    min_distance=min(distance)
    List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance))]
    Kpoint_path=[]
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[0]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(int(distance[1]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(M),str(int(distance[2]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[3]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(R),str(int(distance[4]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(0)))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(M),str(int(distance[5]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(R),str(0)))
    data=[List,List_distance]
    return Kpoint_path,data

###k-point path for face-centered cubic
def Kpoint_fcc(factor):
    #Gamma-X-W-K-Gamma-L-U-W-L-K|U-X
    List=['$\Gamma$','$X$','$W$','$K$','$\Gamma$','$L$','$U$','$W$','$L$','$K$|$U$','$X$']
    distance=[]
    Gamma=[0,0,0]
    K=[0.375,0.375,0.75]
    X=[0.5,0,0.5]
    L=[0.5,0.5,0.5]
    U=[0.625,0.25,0.625]
    W=[0.5,0.25,0.75]
    distance.append(Length(Gamma,X))
    distance.append(Length(X,W))
    distance.append(Length(W,K))
    distance.append(Length(K,Gamma))
    distance.append(Length(Gamma,L))
    distance.append(Length(L,U))
    distance.append(Length(U,W))
    distance.append(Length(W,L))
    distance.append(Length(L,K))
    distance.append(Length(U,X))
    min_distance=min(distance)
    List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance))]
    Kpoint_path=[]
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[0]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(int(distance[1]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(W),str(int(distance[2]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(K),str(int(distance[3]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[4]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(L),str(int(distance[5]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(U),str(int(distance[6]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(W),str(int(distance[7]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(L),str(int(distance[8]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(K),str(0)))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(U),str(int(distance[9]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(0)))
    data=[List,List_distance]
    return Kpoint_path,data

###k-point path for body-centered cubic
def Kpoint_bcc(factor):
    #Gamma-H-N-Gamma-P-H|P-N
    List=['$\Gamma$','$H$','$N$','$\Gamma$','$P$','$H$|$P$','$N$']
    distance=[]
    Gamma=[0,0,0]
    H=[0.5,-0.5,0.5]
    P=[0.25,0.25,0.25]
    N=[0,0,0.5]
    distance.append(Length(Gamma,H))
    distance.append(Length(H,N))
    distance.append(Length(N,Gamma))
    distance.append(Length(Gamma,P))
    distance.append(Length(P,H))
    distance.append(Length(P,N))
    min_distance=min(distance)
    List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance))]
    Kpoint_path=[]
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[0]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(H),str(int(distance[1]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(N),str(int(distance[2]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[3]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(P),str(int(distance[4]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(H),str(0)))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(P),str(int(distance[5]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(N),str(0)))
    data=[List,List_distance]
    return Kpoint_path,data

###k-point path for primitive tetragonal
def Kpoint_tetragonal(factor):
    #Gamma-X-M-Gamma-Z-R-A-Z|X-R|M-A
    List=['$\Gamma$','$X$','$M$','$\Gamma$','$Z$','$R$','$A$','$Z$|$X$','$R$|$M$','$A$']
    distance=[]
    Gamma=[0,0,0]
    A=[0.5,0.5,0.5]
    M=[0.5,0.5,0]
    R=[0,0.5,0.5]
    X=[0,0.5,0]
    Z=[0,0,0.5]
    distance.append(Length(Gamma,X))
    distance.append(Length(X,M))
    distance.append(Length(M,Gamma))
    distance.append(Length(Gamma,Z))
    distance.append(Length(Z,R))
    distance.append(Length(R,A))
    distance.append(Length(A,Z))
    distance.append(Length(X,R))
    distance.append(Length(M,A))
    min_distance=min(distance)
    List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance))]
    Kpoint_path=[]
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[0]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(int(distance[1]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(M),str(int(distance[2]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[3]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(int(distance[4]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(R),str(int(distance[5]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(A),str(int(distance[6]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(0)))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(int(distance[7]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(R),str(0)))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(M),str(int(distance[8]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(A),str(0)))
    data=[List,List_distance]
    return Kpoint_path,data

###k-point path for body-centered tetragonal
def Kpoint_body_tetragonal(factor,celldm_3):
    distance=[]
    Gamma=[0,0,0]
    N=[0,0.5,0]
    P=[0.25,0.25,0.25]
    X=[0,0,0.5]
    eta=(1+celldm_3**2)/4.0
    psi=0.5*celldm_3**2
    if celldm_3 < 1.0:
        #Gamma-X-M-Gamma-Z-P-N-Z1-M|X-P
        List=['$\Gamma$','$X$','$M$','$\Gamma$','$Z$','$P$','$N$','$Z_1$','$M$|$X$','$P$']
        M=[-0.5,0.5,0.5]
        Z=[eta,eta,-eta]
        Z1=[-eta,1-eta,eta]
        distance.append(Length(Gamma,X))
        distance.append(Length(X,M))
        distance.append(Length(M,Gamma))
        distance.append(Length(Gamma,Z))
        distance.append(Length(Z,P))
        distance.append(Length(P,N))
        distance.append(Length(N,Z1))
        distance.append(Length(Z1,M))
        distance.append(Length(X,P))
        min_distance=min(distance)
        List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance))]
        Kpoint_path=[]
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[0]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(int(distance[1]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(M),str(int(distance[2]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[3]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(int(distance[4]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(P),str(int(distance[5]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(N),str(int(distance[6]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z1),str(int(distance[7]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(M),str(0)))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(int(distance[8]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(P),str(0)))
    else:
        #Gamma-X-Y-Sigma-Gamma-Z-Sigma1-N-P-Y1-Z|X-P
        List=['$\Gamma$','$X$','$Y$','$\Sigma$','$\Gamma$','$Z$','$\Sigma_1$','$N$','$P$','$Y_1$','$Z$|$X$','$P$']
        Sigma=[-eta,eta,eta]
        Sigma1=[eta,1-eta,-eta]
        Y=[-psi,psi,0.5]
        Y1=[0.5,0.5,-psi]
        Z=[0.5,0.5,-0.5]
        distance.append(Length(Gamma,X))
        distance.append(Length(X,Y))
        distance.append(Length(Y,Sigma))
        distance.append(Length(Sigma,Gamma))
        distance.append(Length(Gamma,Z))
        distance.append(Length(Z,Sigma1))
        distance.append(Length(Sigma1,N))
        distance.append(Length(N,P))
        distance.append(Length(P,Y1))
        distance.append(Length(Y1,Z))
        distance.append(Length(X,P))
        min_distance=min(distance)
        List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance))]
        Kpoint_path=[]
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[0]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(int(distance[1]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y),str(int(distance[2]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Sigma),str(int(distance[3]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[4]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(int(distance[5]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Sigma1),str(int(distance[6]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(N),str(int(distance[7]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(P),str(int(distance[8]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y1),str(int(distance[9]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(0)))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(int(distance[10]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(P),str(0)))
    data=[List,List_distance]
    return Kpoint_path,data

###k-point path for primitive orthorhombic
def Kpoint_orthorhombic(factor):
    #Gamma-X-S-Y-Gamma-Z-U-R-T-Z|Y-T|U-X|S-R
    List=['$\Gamma$','$X$','$S$','$Y$','$\Gamma$','$Z$','$U$','$R$','$T$','$Z$|$Y$','$T$|$U$','$X$|$S$','$P$']
    distance=[]
    Gamma=[0,0,0]
    R=[0.5,0.5,0.5]
    S=[0.5,0.5,0]
    T=[0,0.5,0.5]
    U=[0.5,0,0.5]
    X=[0.5,0,0]
    Y=[0,0.5,0]
    Z=[0,0,0.5]
    distance.append(Length(Gamma,X))
    distance.append(Length(X,S))
    distance.append(Length(S,Y))
    distance.append(Length(Y,Gamma))
    distance.append(Length(Gamma,Z))
    distance.append(Length(Z,U))
    distance.append(Length(U,R))
    distance.append(Length(R,T))
    distance.append(Length(T,Z))
    distance.append(Length(Y,T))
    distance.append(Length(U,X))
    distance.append(Length(S,R))
    min_distance=min(distance)
    List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance))]
    Kpoint_path=[]
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[0]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(int(distance[1]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(S),str(int(distance[2]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y),str(int(distance[3]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[4]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(int(distance[5]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(U),str(int(distance[6]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(R),str(int(distance[7]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(T),str(int(distance[8]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(0)))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y),str(int(distance[9]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(T),str(0)))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(U),str(int(distance[10]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(0)))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(S),str(int(distance[11]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(R),str(0)))
    data=[List,List_distance]
    return Kpoint_path,data

###k-point path for face-centered orthorhombic
def Kpoint_face_orthorhombic(factor,celldm_1,celldm_2,celldm_3):
    distance=[]
    a,b,c=order_parameters(celldm_1,celldm_2,celldm_3)
    Gamma=[0,0,0]
    L=[0.5,0.5,0.5]
    Y=[0.5,0.0,0.5]
    Z=[0.5,0.5,0]
    eta=(1+a**2/b**2-a**2/c**2)/4.0
    if 1/b**2+1/c**2 > 1/a**2:
        #Gamma-Y-C-D-X-Gamma-Z-D1-H-C|C1-Z|X-H1|H-Y|L-Gamma
        List=['$\Gamma$','$Y$','$C$','$D$','$X$','$\Gamma$','$Z$','$D_1$','$H$','$C$|$C_1$','$Z$|$X$','$H_1$|$H$','$Y$|$L$','$\Gamma$']
        delta=(1+(b/a)**2-(b/c)**2)/4.0
        phi=(1+(c/b)**2-(c/a)**2)/4.0
        C=[0.5,0.5-eta,1-eta]
        C1=[0.5,0.5+eta,eta]
        D=[0.5-delta,0.5,1-delta]
        D1=[0.5+delta,0.5,delta]
        H=[1-phi,0.5-phi,0.5]
        H1=[phi,0.5+phi,0.5]
        X=[0,0.5,0]
        distance.append(Length(Gamma,Y))
        distance.append(Length(Y,C))
        distance.append(Length(C,D))
        distance.append(Length(D,X))
        distance.append(Length(X,Gamma))
        distance.append(Length(Gamma,Z))
        distance.append(Length(Z,D1))
        distance.append(Length(D1,H))
        distance.append(Length(H,C))
        distance.append(Length(C1,Z))
        distance.append(Length(X,H1))
        distance.append(Length(H,Y))
        distance.append(Length(L,Gamma))
        min_distance=min(distance)
        List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance))]
        Kpoint_path=[]
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[0]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y),str(int(distance[1]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(C),str(int(distance[2]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(D),str(int(distance[3]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(int(distance[4]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[5]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(int(distance[6]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(D1),str(int(distance[7]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(H),str(int(distance[8]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(C),str(0)))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(C1),str(int(distance[9]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(0)))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(int(distance[10]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(H1),str(0)))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(H),str(int(distance[11]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y),str(0)))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(L),str(int(distance[12]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(0)))
    else:
        #Gamma-Y-T-Z-Gamma-X-A1-Y(|T-X1|)X-A-Z|L-Gamma
        List=['$\Gamma$','$Y$','$T$','$Z$','$\Gamma$','$X$','$A_1$','$Y$|$X$','$A$','$Z$|$L$','$\Gamma$']
        eps=(1+(a/b)**2+(a/c)**2)/4.0
        A=[0.5,0.5+eps,eps]
        A1=[0.5,0.5-eps,1-eps]
        T=[1,0.5,0.5]
        X=[0,eta,eta]
        X1=[1,1-eta,1-eta]
        distance.append(Length(Gamma,Y))
        distance.append(Length(Y,T))
        distance.append(Length(T,Z))
        distance.append(Length(Z,Gamma))
        distance.append(Length(Gamma,X))
        distance.append(Length(X,A1))
        distance.append(Length(A1,Y))
        distance.append(Length(T,X1))
        distance.append(Length(X,A))
        distance.append(Length(A,Z))
        distance.append(Length(L,Gamma))
        min_distance=min(distance)
        List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance)) if i!=7]
        Kpoint_path=[]
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[0]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y),str(int(distance[1]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(T),str(int(distance[2]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(int(distance[3]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[4]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(int(distance[5]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(A1),str(int(distance[6]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y),str(0)))
        if (1/c)**2+1/b**2 < 1/a**2:
            List=['$\Gamma$','$Y$','$T$','$Z$','$\Gamma$','$X$','$A_1$','$Y$|$T$','$X_1$|$X$','$A$','$Z$|$L$','$\Gamma$']
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(T),str(int(distance[7]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X1),str(0)))
            List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance))]
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(int(distance[8]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(A),str(int(distance[9]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(0)))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(L),str(int(distance[10]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(0)))
    data=[List,List_distance]
    return Kpoint_path,data

###k-point path for body-centered orthorhombic
def Kpoint_body_orthorhombic(factor,celldm_1,celldm_2,celldm_3):
    a,b,c=order_parameters(celldm_1,celldm_2,celldm_3)
    #Gamma-X-L-T-W-R-X1-Z-Gamma-Y-S-W|L1-Y|Y1-Z
    List=['$\Gamma$','$X$','$L$','$T$','$W$','$R$','$X_1$','$Z$','$\Gamma$','$Y$','$S$','$W$|$L_1$','$Y$|$Y_1$','$Z$']
    psi=(1+(a/c)**2)/4.0
    eta=(1+(b/c)**2)/4.0
    delta=((b/c)**2-(a/c)**2)/4.0
    mu=((a/c)**2+(b/c)**2)/4.0
    distance=[]
    Gamma=[0,0,0]
    L=[-mu,mu,0.5-delta]
    L1=[mu,-mu,0.5+delta]
    L2=[0.5-delta,0.5+delta,-mu]
    R=[0,0.5,0]
    S=[0.5,0,0]
    T=[0,0,0.5]
    W=[0.25,0.25,0.25]
    X=[-psi,psi,psi]
    X1=[psi,1-psi,-psi]
    Y=[eta,-eta,eta]
    Y1=[1-eta,eta,-eta]
    Z=[0.5,0.5,-0.5]
    distance.append(Length(Gamma,X))
    distance.append(Length(X,L))
    distance.append(Length(L,T))
    distance.append(Length(T,W))
    distance.append(Length(W,R))
    distance.append(Length(R,X1))
    distance.append(Length(X1,Z))
    distance.append(Length(Z,Gamma))
    distance.append(Length(Gamma,Y))
    distance.append(Length(Y,S))
    distance.append(Length(S,W))
    distance.append(Length(L1,Y))
    distance.append(Length(Y1,Z))
    min_distance=min(distance)
    List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance))]
    Kpoint_path=[]
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[0]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(int(distance[1]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(L),str(int(distance[2]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(T),str(int(distance[3]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(W),str(int(distance[4]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(R),str(int(distance[5]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X1),str(int(distance[6]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(int(distance[7]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[8]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y),str(int(distance[9]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(S),str(int(distance[10]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(W),str(0)))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(L1),str(int(distance[11]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y),str(0)))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y1),str(int(distance[12]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(0)))
    data=[List,List_distance]
    return Kpoint_path,data

###k-point path for c-centered orthorhombic
def Kpoint_c_orthorhombic(factor,celldm_1,celldm_2,celldm_3):
    a,b,c=order_parameters(celldm_1,celldm_2,celldm_3)
    #Gamma-X-S-R-A-Z-Gamma-Y-X1-A1-T-Y|Z-T
    List=['$\Gamma$','$X$','$S$','$R$','$A$','$Z$','$\Gamma$','$Y$','$X_1$','$A_1$','$T$','$Y$|$Z$','$T$']
    psi=(1+(a/b)**2)/4.0
    distance=[]
    Gamma=[0,0,0]
    A=[psi,psi,0.5]
    A1=[-psi,1-psi,0.5]
    R=[0,0.5,0.5]
    S=[0,0.5,0]
    T=[-0.5,0.5,0.5]
    X=[psi,psi,0]
    X1=[-psi,1-psi,0]
    Y=[-0.5,0.5,0]
    Z=[0,0,0.5]
    distance.append(Length(Gamma,X))
    distance.append(Length(X,S))
    distance.append(Length(S,R))
    distance.append(Length(R,A))
    distance.append(Length(A,Z))
    distance.append(Length(Z,Gamma))
    distance.append(Length(Gamma,Y))
    distance.append(Length(Y,X1))
    distance.append(Length(X1,A1))
    distance.append(Length(A1,T))
    distance.append(Length(T,Y))
    distance.append(Length(Z,T))
    min_distance=min(distance)
    List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance))]
    Kpoint_path=[]
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[0]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(int(distance[1]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(S),str(int(distance[2]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(R),str(int(distance[3]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(A),str(int(distance[4]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(int(distance[5]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[6]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y),str(int(distance[7]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X1),str(int(distance[8]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(A1),str(int(distance[9]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(T),str(int(distance[10]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y),str(0)))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(int(distance[11]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(T),str(0)))
    data=[List,List_distance]
    return Kpoint_path,data

###k-point path for hexagonal
def Kpoint_hexagonal(factor,celldm_3):
    #Gamma-M-K-Gamma-A-L-H-A|L-M|K-H
    List=['$\Gamma$','$M$','$K$','$\Gamma$','$A$','$L$','$H$','$A$|$L$','$M$|$K$','$H$']
    distance=[]
    Gamma=[0,0,0]
    A=[0,0,0.5]
    H=[0.33333,0.33333,0.5]
    K=[0.33333,0.33333,0]
    L=[0.5,0,0.5]
    M=[0.5,0,0]
    distance.append(Length(Gamma,M))
    distance.append(Length(M,K))
    distance.append(Length(K,Gamma))
    distance.append(Length(Gamma,A))
    distance.append(Length(A,L))
    distance.append(Length(L,H))
    distance.append(Length(H,A))
    distance.append(Length(L,M))
    distance.append(Length(K,H))
    min_distance=min(distance)
    List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance))]
    Kpoint_path=[]
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[0]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(M),str(int(distance[1]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(K),str(int(distance[2]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[3]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(A),str(int(distance[4]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(L),str(int(distance[5]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(H),str(int(distance[6]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(A),str(0)))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(L),str(int(distance[7]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(M),str(0)))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(K),str(int(distance[8]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(H),str(0)))
    data=[List,List_distance]
    return Kpoint_path,data

###k-point path for rhombohedral
def Kpoint_rhombohedral(factor,alpha):
    if alpha < 90:
        #Gamma-L-B1|B-Z-Gamma-X|Q-F-P1-Z|L-P
        List=['$\Gamma$','$L$','$B_1$|$B$','$Z$','$\Gamma$','$X$|$Q$','$F$','$P_1$','$Z$|$L$','$P$']
        eta=(1+4*np.cos(alpha*np.pi/180.))/(2+4*np.cos(alpha*np.pi/180.))
        nu=0.75-eta/2.
        distance=[]
        Gamma=[0,0,0]
        B=[eta,0.5,1-eta]
        B1=[0.5,1-eta,eta-1]
        F=[0.5,0.5,0]
        L=[0.5,0,0]
        L1=[0,0,-0.5]
        P=[eta,nu,nu]
        P1=[1-nu,1-nu,1-eta]
        P2=[nu,nu,1-eta]
        Q=[1-nu,nu,0]
        X=[nu,0,-nu]
        Z=[0.5,0.5,0.5]
        distance.append(Length(Gamma,L))
        distance.append(Length(L,B1))
        distance.append(Length(B,Z))
        distance.append(Length(Z,Gamma))
        distance.append(Length(Gamma,X))
        distance.append(Length(Q,F))
        distance.append(Length(F,P1))
        distance.append(Length(P1,Z))
        distance.append(Length(L,P))
        min_distance=min(distance)
        List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance))]
        Kpoint_path=[]
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[0]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(L),str(int(distance[1]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(B1),str(0)))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(B),str(int(distance[2]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(int(distance[3]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[4]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(0)))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Q),str(int(distance[5]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(F),str(int(distance[6]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(P1),str(int(distance[7]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(0)))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(L),str(int(distance[8]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(P),str(0)))
    else:
        #Gamma-P-Z-Q-Gamma-F-P1-Q1-L-Z
        List=['$\Gamma$','$P$','$Z$','$Q$','$\Gamma$','$F$','$P_1$','$Q_1$','$L$','$Z$']
        eta=1/(2*(np.tan(alpha*np.pi/360.))**2)
        nu=0.75-eta/2.
        distance=[]
        Gamma=[0,0,0]
        F=[0.5,-0.5,0]
        L=[0.5,0,0]
        P=[1-nu,-nu,1-nu]
        P1=[nu,nu-1,nu-1]
        Q=[eta,eta,eta]
        Q1=[1-eta,-eta,-eta]
        Z=[0.5,-0.5,0.5]
        distance.append(Length(Gamma,P))
        distance.append(Length(P,Z))
        distance.append(Length(Z,Q))
        distance.append(Length(Q,Gamma))
        distance.append(Length(Gamma,F))
        distance.append(Length(F,P1))
        distance.append(Length(P1,Q1))
        distance.append(Length(Q1,L))
        distance.append(Length(L,Z))
        min_distance=min(distance)
        List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance))]
        Kpoint_path=[]
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[0]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(P),str(int(distance[1]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(int(distance[2]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Q),str(int(distance[3]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[4]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(F),str(int(distance[5]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(P1),str(int(distance[6]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Q1),str(int(distance[7]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(L),str(int(distance[8]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(0)))
    data=[List,List_distance]
    return Kpoint_path,data

###k-point path for primitive monoclinic
def Kpoint_monoclinic(factor,celldm_1,celldm_2,celldm_3,gamma):
    #Gamma-Y-H-C-E-M1-A-X-H1|M-D-Z|Y-D
    List=['$\Gamma$','$Y$','$H$','$C$','$E$','$M_1$','$A$','$X$','$H_1$|$M$','$D$','$Z$|$Y$','$D$']
    distance=[]
    c,a,b=order_parameters(celldm_1,celldm_2,celldm_3) #different order because the paper uses alpha as not 90 deg while QE uses gamma
    eta=(1-b/c*np.cos(gamma*np.pi/180.)/(2*(np.sin(gamma*np.pi/180.))**2))
    nu=0.5-eta*c/b*np.cos(gamma*np.pi/180.)
    Gamma=[0,0,0]
    A=[0.5,0.5,0]
    C=[0,0.5,0.5]
    D=[0.5,0,0.5]
    D1=[0.5,0,-0.5]
    E=[0.5,0.5,0.5]
    H=[0,eta,1-nu]
    H1=[0,1-eta,nu]
    H2=[0,eta,-nu]
    M=[0.5,eta,1-nu]
    M1=[0.5,1-eta,nu]
    M2=[0.5,eta,-nu]
    X=[0,0.5,0]
    Y=[0,0,0.5]
    Y1=[0,0,-0.5]
    Z=[0.5,0,0]
    distance.append(Length(Gamma,Y))
    distance.append(Length(Y,H))
    distance.append(Length(H,C))
    distance.append(Length(C,E))
    distance.append(Length(E,M1))
    distance.append(Length(M1,A))
    distance.append(Length(A,X))
    distance.append(Length(X,H1))
    distance.append(Length(M,D))
    distance.append(Length(D,Z))
    distance.append(Length(Y,D))
    min_distance=min(distance)
    List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance))]
    Kpoint_path=[]
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[0]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y),str(int(distance[1]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(H),str(int(distance[2]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(C),str(int(distance[3]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(E),str(int(distance[4]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(M1),str(int(distance[5]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(A),str(int(distance[6]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(int(distance[7]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(H1),str(0)))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(M),str(int(distance[8]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(D),str(int(distance[9]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(0)))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y),str(int(distance[10]/min_distance*factor))))
    Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(D),str(0)))
    data=[List,List_distance]
    return Kpoint_path,data

###k-point path for c-centered monoclinic
def Kpoint_c_monoclinic(factor,celldm_1,celldm_2,celldm_3,gamma):
    c,a,b=order_parameters(celldm_1,celldm_2,celldm_3) #different order because the paper uses alpha as not 90 deg while QE uses gamma
    distance=[]
    if b <= a*np.sin(np.radians(gamma)):
        #Gamma-Y-F-L-I|I1-Z-F1|Y-X1|X-Gamma-N|M-Gamma
        #Gamma-Y-F-L-I|I1-Z-F1|Y-X1|N-Gamma-M
        xi=(2-b/c*np.cos(gamma*np.pi/180))/(4*(np.sin(np.radians(gamma)))**2)
        eta=0.5+2*xi*b/c*np.cos(np.radians(gamma))
        psi=0.75-a**2/(4*b**2*(np.sin(np.radians(gamma)))**2)
        phi=psi+(0.75-psi)*b/c*np.cos(np.radians(gamma))
        Gamma=[0,0,0]
        N=[0.5,0,0]
        N1=[0,-0.5,0]
        F=[1-xi,1-xi,1-eta]
        F1=[xi,xi,eta]
        F2=[-xi,-xi,1-eta]
        F3=[1-xi,-xi,1-eta]
        I=[phi,1-phi,0.5]
        I1=[1-phi,phi-1,0.5]
        L=[0.5,0.5,0.5]
        M=[0.5,0,0.5]
        X=[1-psi,psi-1,0]
        X1=[psi,1-psi,0]
        X2=[psi-1,-psi,0]
        Y=[0.5,0.5,0]
        Y1=[-0.5,-0.5,0]
        Z=[0,0,0.5]
        distance.append(Length(Gamma,Y))
        distance.append(Length(Y,F))
        distance.append(Length(F,L))
        distance.append(Length(L,I))
        distance.append(Length(I1,Z))
        distance.append(Length(Z,F1))
        distance.append(Length(Y,X1))
        distance.append(Length(X,Gamma))
        distance.append(Length(Gamma,N))
        distance.append(Length(M,Gamma))
        distance.append(Length(N,Gamma))
        distance.append(Length(Gamma,M))
        min_distance=min(distance)
        Kpoint_path=[]
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[0]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y),str(int(distance[1]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(F),str(int(distance[2]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(L),str(int(distance[3]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(I),str(0)))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(I1),str(int(distance[4]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(int(distance[5]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(F1),str(0)))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y),str(int(distance[6]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X1),str(0)))
        List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance)) if i < 7]
        if b < a*np.sin(np.radians(gamma)):
            for i in range(7,10):
                List_distance.append(int(distance[i]/min_distance*factor))
            List=['$\Gamma$','$Y$','$F$','$L$','$I$|$I_1$','$Z$','$F_1$|$Y$','$X_1$|$X$','$\Gamma$','$N$|$M$','$\Gamma$']
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(int(distance[7]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[8]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(N),str(0)))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(M),str(int(distance[9]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(0)))
        else:
            for i in range(10,12):
                List_distance.append(int(distance[i]/min_distance*factor))
            List=['$\Gamma$','$Y$','$F$','$L$','$I$|$I_1$','$Z$','$F_1$|$Y$','$X_1$|$N$','$\Gamma$','$M$']
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(N),str(int(distance[10]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[11]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(M),str(0)))
    elif b > a*np.sin(np.radians(gamma)):
        if (b/c*np.cos(np.radians(gamma))+(b/a*np.cos(np.radians(gamma)))**2) <= 1:
            #Gamma-Y-F-H-Z-I-F1|H1-Y1-X-Gamma-N|M-Gamma
            #Gamma-Y-F-H-Z-I|H1-Y1-X-Gamma-N|M-Gamma
            mu=(1+(b/a)**2)/4.0
            delta=b*c*np.cos(np.radians(gamma))/(2*a**2)
            xi=mu-0.25+(1-b/c*np.cos(np.radians(gamma)))/(4*(np.sin(np.radians(gamma)))**2)
            eta=0.5+2*xi*c/b*np.cos(np.radians(gamma))
            phi=1+xi-2*mu
            psi=eta-2*delta
            Gamma=[0,0,0]
            F=[1-phi,1-phi,1-psi]
            F1=[phi,phi-1,psi]
            F2=[1-phi,-phi,1-psi]
            H=[xi,xi,eta]
            H1=[1-xi,-xi,1-eta]
            H2=[-xi,-xi,1-eta]
            I=[0.5,-0.5,0.5]
            M=[0.5,0,0.5]
            N=[0.5,0,0]
            N1=[0,-0.5,0]
            X=[0.5,-0.5,0]
            Y=[mu,mu,delta]
            Y1=[1-mu,-mu,-delta]
            Y2=[-mu,-mu,-delta]
            Y3=[mu,mu-1,delta]
            Z=[0,0,0.5]
            distance.append(Length(Gamma,Y))
            distance.append(Length(Y,F))
            distance.append(Length(F,H))
            distance.append(Length(H,Z))
            distance.append(Length(Z,I))
            distance.append(Length(I,F1))
            distance.append(Length(H1,Y1))
            distance.append(Length(Y1,X))
            distance.append(Length(X,Gamma))
            distance.append(Length(Gamma,N))
            distance.append(Length(M,Gamma))
            min_distance=min(distance)
            Kpoint_path=[]
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[0]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y),str(int(distance[1]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(F),str(int(distance[2]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(H),str(int(distance[3]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(int(distance[4]/min_distance*factor))))
            if (b/c*np.cos(np.radians(gamma))+(b/a*np.cos(np.radians(gamma)))**2) < 1:
                List=['$\Gamma$','$Y$','$F$','$H$','$Z$','$I$','$F_1$|$H_1$','$Y_1$','$X$','$\Gamma$','$N$|$M$','$\Gamma$']
                List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance))]
                Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(I),str(int(distance[5]/min_distance*factor))))
                Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(F1),str(0)))
            else:
                List=['$\Gamma$','$Y$','$F$','$H$','$Z$','$I$|$H_1$','$Y_1$','$X$','$\Gamma$','$N$|$M$','$\Gamma$']
                List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance)) if i != 5]
                Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(I),str(0)))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(H1),str(int(distance[6]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y1),str(int(distance[7]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(int(distance[8]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[9]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(N),str(0)))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(M),str(int(distance[10]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(0)))
        else:
            #Gamma-Y-F-L-I|I1-Z-H-F1|H1-Y1-X-Gamma-N|M-Gamma
            List=['$\Gamma$','$Y$','$F$','$H$','$Z$','$I$|$H_1$','$Y_1$','$X$','$\Gamma$','$N$|$M$','$\Gamma$']
            xi=((b/a)**2+(1-b/c*np.cos(np.radians(gamma)))/(np.sin(np.radians(gamma)))**2)/4.0
            eta=0.5+2*xi*c/b*np.cos(np.radians(gamma))
            mu=eta/2.+(b/(2*a))**2-b*c*np.cos(np.radians(gamma))/(2*a**2)
            nu=2*mu-eta
            rho=1-xi*(a/b)**2
            omega=(4*nu-1-(b/a*np.sin(np.radians(gamma)))**2)*c/(2*b*np.cos(np.radians(gamma)))
            delta=xi*c/b*np.cos(np.radians(gamma))+omega/2-0.25
            Gamma=[0,0,0]
            F=[nu,nu,omega]
            F1=[1-nu,1-nu,1-omega]
            F2=[nu,nu-1,omega]
            H=[xi,xi,eta]
            H1=[1-xi,-xi,1-eta]
            H2=[-xi,-xi,1-eta]
            I=[rho,1-rho,0.5]
            I1=[1-rho,rho-1,0.5]
            L=[0.5,0.5,0.5]
            M=[0.5,0,0.5]
            N=[0.5,0,0]
            N1=[0,-0.5,0]
            X=[0.5,-0.5,0]
            Y=[mu,mu,delta]
            Y1=[1-mu,-mu,-delta]
            Y2=[-mu,-mu,-delta]
            Y3=[mu,mu-1,delta]
            Z=[0,0,0.5]
            distance.append(Length(Gamma,Y))
            distance.append(Length(Y,F))
            distance.append(Length(F,L))
            distance.append(Length(L,I))
            distance.append(Length(I1,Z))
            distance.append(Length(Z,H))
            distance.append(Length(H,F1))
            distance.append(Length(H1,Y1))
            distance.append(Length(Y1,X))
            distance.append(Length(X,Gamma))
            distance.append(Length(Gamma,N))
            distance.append(Length(M,Gamma))
            min_distance=min(distance)
            List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance))]
            Kpoint_path=[]
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[0]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y),str(int(distance[1]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(F),str(int(distance[2]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(L),str(int(distance[3]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(I),str(0)))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(I1),str(int(distance[4]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(int(distance[5]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(H),str(int(distance[6]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(F1),str(0)))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(H1),str(int(distance[7]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y1),str(int(distance[8]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(int(distance[9]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[10]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(N),str(0)))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(M),str(int(distance[11]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(0)))
    data=[List,List_distance]
    return Kpoint_path,data

###k-point path for triclinic
def Kpoint_triclinic(factor,celldm_1,celldm_2,celldm_3,alpha,beta,gamma):
    distance=[]
    kalpha=np.arccos((np.cos(np.radians(beta))*np.cos(np.radians(gamma))-np.cos(np.radians(alpha)))/(np.sin(np.radians(beta))*np.sin(np.radians(gamma))))
    kbeta=np.arccos((np.cos(np.radians(alpha))*np.cos(np.radians(gamma))-np.cos(np.radians(beta)))/(np.sin(np.radians(alpha))*np.sin(np.radians(gamma))))
    kgamma=np.arccos((np.cos(np.radians(beta))*np.cos(np.radians(alpha))-np.cos(np.radians(gamma)))/(np.sin(np.radians(beta))*np.sin(np.radians(alpha))))
    if kalpha <np.radians(90):
        if kbeta < np.radians(90):
            #X-Gamma-Y|L-Gamma-Z|N-Gamma-M|R-Gamma  
            List=['X','$\Gamma$','$Y$|$L$','$\Gamma$','$Z$|$N$','$\Gamma$','$M$|$R$','$\Gamma$']
            Gamma=[0,0,0]
            L=[0.5,0.5,0]
            M=[0,0.5,0.5]
            N=[0.5,0,0.5]
            R=[0.5,0.5,0.5]
            X=[0.5,0,0]
            Y=[0,0.5,0]
            Z=[0,0,0.5]
            distance.append(Length(X,Gamma))
            distance.append(Length(Gamma,Y))
            distance.append(Length(L,Gamma))
            distance.append(Length(Gamma,Z))
            distance.append(Length(N,Gamma))
            distance.append(Length(Gamma,M))
            distance.append(Length(R,Gamma))
            min_distance=min(distance)
            List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance))]
            Kpoint_path=[]
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(int(distance[0]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[1]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y),str(0)))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(L),str(int(distance[2]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[3]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(0)))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(N),str(int(distance[4]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[5]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(M),str(0)))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(R),str(int(distance[6]/min_distance*factor))))
            Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(0)))
    else:
        #X-Gamma-Y|L-Gamma-Z|N-Gamma-M|R-Gamma  
        List=['X','$\Gamma$','$Y$|$L$','$\Gamma$','$Z$|$N$','$\Gamma$','$M$|$R$','$\Gamma$']
        Gamma=[0,0,0]
        L=[0.5,-0.5,0]
        M=[0,0,0.5]
        N=[-0.5,-0.5,0.5]
        R=[0.5,0.5,0.5]
        X=[0,-0.5,0]
        Y=[0.5,0,0]
        Z=[-.5,0,0.5]
        distance.append(Length(X,Gamma))
        distance.append(Length(Gamma,Y))
        distance.append(Length(L,Gamma))
        distance.append(Length(Gamma,Z))
        distance.append(Length(N,Gamma))
        distance.append(Length(Gamma,M))
        distance.append(Length(R,Gamma))
        min_distance=min(distance)
        List_distance=[int(distance[i]/min_distance*factor) for i in range(len(distance))]
        Kpoint_path=[]
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(X),str(int(distance[0]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[1]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Y),str(0)))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(L),str(int(distance[2]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[3]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Z),str(0)))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(N),str(int(distance[4]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(int(distance[5]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(M),str(0)))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(R),str(int(distance[6]/min_distance*factor))))
        Kpoint_path.append('{:.4f} {:.4f} {:.4f}   {}'.format(*tuple(Gamma),str(0)))
    data=[List,List_distance]
    return Kpoint_path,data



def Bravais_lattice(Space_group,Bravais_group):
    if Space_group<3:
        Bravais=14
    elif Space_group <16:
        if Bravais_group == 'P':
            Bravais =12
        else:
            Bravais =13
    elif Space_group <75:
        if Bravais_group == 'P':
            Bravais = 8
        elif Bravais_group == 'I':
            Bravais = 11
        elif Bravais_group == 'F':
            Bravais = 10
        else:
            Bravais = 9
    elif Space_group < 195:
        if Bravais_group == 'P':
            Bravais = 4
        else:
            Bravais = 5
    elif Space_group <231:
        if Bravais_group == 'P':
            Bravais = 1
        elif Bravais_group == 'I':
            Bravais = 3
        elif Bravais_group == 'F':
            Bravais = 2
    return Bravais
#=========================================================
# End of Kpoints.py
#=========================================================