#Julian Chaidez
#10.2019

#Trisection Invariant Computation Script v3

#This Python script is written to calculate the Trisection-Kuperberg invariant
#in the paper [insert arxiv link].

#IMPORTED MODULES

import numpy as np 

#SPARSE TENSOR ALGEBRA CODE

#Tuple Utilities

def excise(A,I):

    B = []
    for j in range(len(A)):
        if not(j in I):
            B.append(A[j])

    return tuple(B)

def intersect(A,B):

    C = []
    for a in A:
        if a in B:
            C.append(a)

    return C

#SparseTensor class.

class SparseTensor:

    #SparseTensor is an implementation of sparse tensors as dictionary objects.

    def __init__(self,entries={():1.0},labels=(),shape=()):

        if type(labels) != tuple:
            raise Exception("type(labels) != tuple")
        if type(shape) != tuple:
            raise Exception("type(shape) != tuple")           

        self.labels = labels
        self.shape = shape
        self.entries = entries

    def __getitem__(self,I):

        if I in self.entries.keys(): return self.entries[I]
        else: return 0.0

    def __setitem__(self,I,v):

        self.entries[I] = v

    def __add__(self,T):

        if not(self.labels == T.labels) and not(self.shape == T.shape):
            raise Exception("label/shape mismatch: cannot add tensors " + str(self) + " and " + str(T))

        U = SparseTensor({},self.labels,self.shape)

        for I,a in self:
            U[I] = U[I] + a
        for J,b in T:
            U[J] = U[J] + b

        return U

    def __radd__(self,T):

        return self + T

    def __mul__(self,x):

        U = SparseTensor({},self.labels,self.shape)

        for I,a in self:
            U[I] = x*a

        return U

    def __rmul__(self,x):

        return self * x

    def __iter__(self):

        return self.entries.items().__iter__()

    def copy(self):

        return SparseTensor(self.entries.copy(), self.labels, self.shape)

    def get_label_index(self,l):

        m = len(self.labels)
        for i in range(m):
            if self.labels[i] == l:
                return i
        raise Exception(str(l) + " is not a label for " + str(self))

    def relabel(self,L):

        m, labels = len(self.labels), []
        for i in range(m):
            if self.labels[i] in L.keys():
                labels.append(L[self.labels[i]])
            else:
                labels.append(self.labels[i])
        self.labels = labels

#Basic Tensor Operations

def Trace(S,i,j):

    if i == j: raise Exception("can't trace index against itself!")
    if S.shape[i] != S.shape[j]: raise Exception("shapes of indices don't match.")

    lb, sh = excise(S.labels,(i,j)), excise(S.shape,(i,j))
    U = SparseTensor({},lb,sh)

    for I, a in S:
        J = excise(I,(i,j))
        if I[i] == I[j]:
            U[J] = U[J] + a

    return U

def TensorProduct(S,T):

    lb, sh = S.labels + T.labels, S.shape + T.shape
    U = SparseTensor({},lb,sh)

    for I, a in S:
        for J, b in T:
            U[I + J] = a * b

    return U

def Transpose(S,P):

    #input:
    # S - a SparseTensor element.
    # P - a tuple representing a permutation P:{1,..,|P|} -> {1,..,|P|}

    #output:
    # St - S with the indices scrambled by P

    if len(S.shape) != len(P):
        raise Exception("index number and permutation size don't match in Transpose")

    St_elts = {}
    for I, x in S:
        It = tuple([I[i] for i in P])
        St_elts[It] = x
    St_lb = tuple([S.labels[i] for i in P])
    St_sh = tuple([S.shape[i] for i in P])

    St = SparseTensor(St_elts, St_lb, St_sh)

    return St

def IndexMerge(S,P):

    U_elts, U_lb, U_sh = {}, tuple([i for i in range(len(P))]), []

    for I in P:
        
        sh = 1
        for i in I:
            sh *= S.shape[i]
        U_sh.append(sh)

    U_lb, U_sh = tuple(U_lb), tuple(U_sh)

    for J, x in S:

        K = []
        for I in P:

            k, m = 0, len(I)
            for i in range(m):

                n = J[I[i]]
                for j in range(i+1,m):
                    n *= S.shape[I[j]]
                k += n

            K.append(k)

        K = tuple(K)
        U_elts[K] = x

    U = SparseTensor(U_elts, U_lb, U_sh)
    return U

def Concatenate(L,lb = 'a'):

    U_elts, U_lb, U_sh, m = {}, (lb,) + L[0].labels, (len(L),) + L[0].shape, len(L)
    for i in range(m):
        for I, x in L[i]:
            U_elts[(i,) + I] = x

    U = SparseTensor(U_elts, U_lb, U_sh)
    return U

#A fast implementation of contraction
#(which in principle could be done using the above two mathods).

def QuickContract(S,T,A,B,w_labels=False):

    #Takes S and T and contracts 

    #Input:
    #   S, T - a pair of SparseTensor instances.
    #   A - a list of indices (if w_labels = False) or index labels (if w_labels = True) of S
    #   B - a list of indices (if w_labels = False) or index labels (if w_labels = True) of T

    # must have len(A) == len(B).

    #Output:
    #   U - the contraction of S and T along the specified indices.

    if len(A) != len(B):
        raise Exception("number of indices being contracted from in Contract do not match!")

    #maps index labels to numerical index labels if w_labels = True.
    if w_labels:
        numA = [S.get_label_index(a) for a in A]
        numB = [T.get_label_index(b) for b in B]
    else:
        numA = A
        numB = B

    for i in range(len(A)):
        if S.shape[numA[i]] != T.shape[numB[i]]:
            raise Exception("shapes of tensor factors being contracted in QuickContract do not match!")

    #constructs pairs of numerical indices being contracted.
    numAB = [(numA[i],numB[i]) for i in range(len(A))]

    #computes labels and shape of U, and initiates U.
    lb = excise(S.labels, numA) + excise(T.labels,numB)
    sh = excise(S.shape, numA) + excise(T.shape,numB)
    U = SparseTensor({},lb,sh)

    #populates U with entries.
    for I, s in S:
        for J, t in T:

            #checks if I and J are multipliable indices.
            mult_ab = True
            for (a,b) in numAB:
                if I[a] != J[b]:
                    mult_ab = False
                    break

            #if the indices are mutlipliable, adds multiplication to new tensor.
            if mult_ab:
                K = excise(I,numA) + excise(J,numB)
                U[K] = U[K] + s * t
            #else adds nothing.

    #returns U.
    return U

#Basic Tensor Constructions

def DiagonalOnes(k,s):

    #input:
    #k - an int, the number of indices of output.
    #s - an int, the shape parameter of output.

    #output:
    #I - a SparseTensor with k indices, of shape (s,..,s)

    I_elts = {}
    for i in range(s):
        J = tuple([i for j in range(k)])
        I_elts[J] = 1.0
    I_lb, I_sh = tuple([i for i in range(k)]), tuple([s for i in range(k)])

    I = SparseTensor(I_elts, I_lb, I_sh)

    return I

#Sparcity Measures

def log_tensor_size(T):

    S, log_d = T.shape, 0.0
    for s in S:
        log_d += np.log(s)
    return log_d

def log_sparcity(T):

    return np.log(len(T.entries))

def abs_sparcity(T):

    return len(T.entries)

#HOPF ALGEBRA/TRIPLE CODE

#PartHopfAlgebra class

#This class contains just enough of the Hopf algebra data within  

class PartHopfAlgebra:

    def __init__(self,D,e,S,dim):

        #sets structure tensors

        self.D = D
        self.e = e
        self.S = S

        #computes cotrace and cotrace

        self.C = Trace(self.D,0,1)
        self.C.relabel({2:0})

        #sets dimension

        self.dim = dim

    def __call__(self,v):

        return HopfElement(v,self)

    def get_coproduct(self):

        return self.D

    def get_counit(self):

        return self.e

    def get_antipode(self):

        return self.S

    def get_cotrace(self):

        return self.C

    def copy(self):

        return PartHopfAlgebra(self.D.copy(),self.e.copy(),self.S.copy(),self.dim)

#HopfAlgebra class

class HopfAlgebra:

    def __init__(self,M,u,D,e,S,dim):

        #sets structure tensors

        self.M = M
        self.u = u
        self.D = D
        self.e = e
        self.S = S

        #computes cotrace and cotrace

        self.T = Trace(self.M,1,2)
        self.C = Trace(self.D,0,1)
        self.C.relabel({2:0})

        #sets dimension

        self.dim = dim

    def __call__(self,v):

        return HopfElement(v,self)

    def __mul__(self,H):

        return HopfProduct(self,H)

    def __invert__(self):

        return HopfDual(self)

    def get_product(self):

        return self.M

    def get_unit(self):

        return self.u

    def get_coproduct(self):

        return self.D

    def get_counit(self):

        return self.e

    def get_antipode(self):

        return self.S

    def get_trace(self):

        return self.T

    def get_cotrace(self):

        return self.C

    def copy(self):

        return HopfAlgebra(self.M.copy(),self.u.copy(),self.D.copy(),self.e.copy(),self.S.copy(),self.dim)

#HopfElement class
#(abstraction for an element of a Hopf algebra)

class HopfElement:

    def __init__(self,v,H):

        if (type(v) == list) or (type(v) == list):

            if len(v) != H.dim: raise Exception("len(v) != H.dim in HopfElement method.")
            v_elts, v_lb, v_sh = {}, (0,), (len(v),)
            for i in range(len(v)): 
                if v[i] != 0: v_elts[(i,)] = v[i]
            self.values = SparseTensor(v_elts,v_lb,v_sh)

        elif (type(v) == SparseTensor):

            if (len(v.shape) != 1) or (v.shape[0] != H.dim):
                raise Exception("SparseTensor passed to HopfElement is wrong shape.")
            self.values = v

        self.algebra = H

    def __mul__(self,y):

        if self.algebra != y.algebra: raise Exception("self.algebra != y.algebra in __mult__")
        a, b, M = self.values, y.values, self.algebra.M
        c = QuickContract(QuickContract(M,a,(0,),(0,)),b,(0,),(0,))
        return HopfElement(c,self.algebra)

    def __repr__(self):

        return str([self.values[(i,)] for i in range(self.algebra.dim)])

    def __str__(self):

        return str([x for I,x in self.values])

#Basic Constructions & Examples

def HopfDual(H):

    #constructs structure tensors
    Mv = Transpose(H.D,(1,2,0))
    uv = H.e.copy()
    Dv = Transpose(H.M,(2,0,1))
    ev = H.u.copy()
    Sv = Transpose(H.S,(1,0))
    dv = H.dim

    #constructs dual Hopf algebra
    Hv = HopfAlgebra(Mv,uv,Dv,ev,Sv,dv)

    #returns Hv
    return Hv

def HopfProduct(G,H):
    #constructs structure tensors
    MGH = IndexMerge(TensorProduct(G.M,H.M),((0,3),(1,4),(2,5)))
    uGH = IndexMerge(TensorProduct(G.u,H.u),((0,1),))
    DGH = IndexMerge(TensorProduct(G.D,H.D),((0,3),(1,4),(2,5)))
    eGH = IndexMerge(TensorProduct(G.u,H.u),((0,1),))
    SGH = IndexMerge(TensorProduct(G.S,H.S),((0,2),(1,3)))
    dGH = G.dim * H.dim

    #constructs dual Hopf algebra
    GxH = HopfAlgebra(MGH,uGH,DGH,eGH,SGH,dGH)

    #returns GH
    return GxH

def Opp(H):

    #constructs structure tensors
    Mo = Transpose(H.M,(1,0,2))
    uo = H.e.copy()
    Do = H.D.copy()
    eo = H.u.copy()
    So = H.S.copy()
    do = H.dim

    #constructs dual Hopf algebra
    Ho = HopfAlgebra(Mo,uo,Do,eo,So,do)

    #returns Hv
    return Ho

def Cop(H):

    #constructs structure tensors
    Mc = H.M.copy()
    uc = H.e.copy()
    Dc = Transpose(H.D,(0,2,1))
    ec = H.u.copy()
    Sc = H.S.copy()
    dc = H.dim

    #constructs dual Hopf algebra
    Hc = HopfAlgebra(Mc,uc,Dc,ec,Sc,dc)

    #returns Hv
    return Hc

#HopfTriple class

class HopfTriple:

    def __init__(self,algebras,pairings):
        self.algebras = algebras
        self.pairings = [[None,None,None],[None,None,None],[None,None,None]]
        for i in [0,1,2]:
            P = pairings[i]
            self.pairings[i % 3][(i+1) % 3] = SparseTensor(P.entries, (i % 3,(i+1) % 3), P.shape)
            self.pairings[(i+1) % 3][i % 3] = SparseTensor(P.entries, (i % 3,(i+1) % 3), P.shape)

    def __getitem__(self,i):

        if type(i) == int:
            return self.algebras[i % 3]
        else:
            raise Exception("HopfTriple object passed non-int as Hopf algebra index.")

    def __call__(self,i,j):

        if type(i) == int and type(j) == int:
            return self.pairings[i][j]
        else:
            raise Exception("HopfTriple object passed non-int pairing index.")

#Canonical Hopf Triple Constructions.

#The Hopf triple T(H,P) of
#Hopf algebra H and pairing P:H x H -> k.

def QuasiTriangularTriple(H,R):

    #defines Hopf algebras
    H0 = Opp(Cop(HopfDual(H)))
    H1 = Cop(H0)
    H2 = Opp(H0)

    #defines trivial pairings
    P01 = DiagonalOnes(2,H0.dim)
    P20 = DiagonalOnes(2,H2.dim)

    #defines non-trivial pairing
    P12 = R.copy()

    #defines and returns Hopf triple
    HT = HopfTriple([H0,H1,H2],[P01,P12,P20])

    return HT
    
#Hopf Triple Examples:

#Abelian group algebras and triples

def matrix_to_tensor(A):

    stA_elts, stA_lb, stA_sh = {}, (1,2), A.shape

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i][j] != 0:
                stA_elts[(i,j)] = A[i][j]

    return SparseTensor(stA_elts, stA_lb, stA_sh)

def GroupToAlgebra(P):

    #Given the natural permutation rep of a group G, generates the group Hopf algebra.
    #input:
    # P - a list of shape (d,d) np.arrays giving 
    #output:
    # H - a HopfAlgebra instance.

    #constructs product M

    E, d = [], P[0].shape[0]
    Id = np.identity(d)

    for A in P: E.append(matrix_to_tensor(A))

    M   = Concatenate(E,0)

    #constructs unit u

    j_Id = -1
    for i in range(d):
        if np.array_equal(P[i],Id):
            j_Id = i
            break

    if j_Id == -1: raise Exception("no identity element passed to group algebra generator.")
    u = SparseTensor({(j_Id,):1.0},(0,),(d,))

    #constructs coproduct D

    D_elts, D_lb, D_sh = {}, (0,1,2),(d,d,d)
    for i in range(d):
        D_elts[(i,i,i)] = 1.0
    D = SparseTensor(D_elts,D_lb,D_sh)

    #constructs counit e

    e_elts, e_lb, e_sh = {}, (0,),(d,)
    for i in range(d):
        e_elts[(i,)] = 1.0
    e = SparseTensor(e_elts,e_lb,e_sh)

    #constructs antipode S

    S_elts, S_lb, S_sh = {}, (0,1),(d,d)
    for i in range(d):
        for j in range(d):
            if np.array_equal(np.dot(P[i],P[j]),Id):
                S_elts[(i,j)] = 1.0
    S = SparseTensor(S_elts,S_lb,S_sh)

    #constructs Hopf algebra H = \C[G] where G is generated by P.

    H = HopfAlgebra(M,u,D,e,S,d)

    #returns H
    return H

#Example algebras

def CyclicPermutations(n):

    E = np.array([[1.0 if (j == (i+1) % n) else 0.0 for j in range(n)] for i in range(n)])
    P = [np.linalg.matrix_power(E,i) for i in range(n)]
    return P

def CyclicGroupAlgebra(n):

    P = CyclicPermutations(n)
    return GroupToAlgebra(P)

def AbelianGroupAlgebra(P):

    U = CyclicGroupAlgebra(P[0])

    for p in P[1:]:

        U = U * CyclicGroupAlgebra(p)

    return U

def DihedralPermutations(n):

    def r(i,j):
        if (i < n) and (j < n):
            return j == (i+1) % n
        elif (i >= n) and (j >= n):
            return (i - n) == (j - n + 1) % n
        else:
            return False

    def s(i,j):
        return j == (i + n) % (2*n)

    R = np.array([[1.0 if r(i,j) else 0.0 for j in range(2*n)] for i in range(2*n)])
    S = np.array([[1.0 if s(i,j) else 0.0 for j in range(2*n)] for i in range(2*n)])

    P = [np.linalg.matrix_power(R,i) for i in range(n)] + [np.dot(np.linalg.matrix_power(R,i),S) for i in range(n)]
    return P 

def DihedralGroupAlgebra(n):

    P = DihedralPermutations(n)
    return GroupToAlgebra(P)

#Example triples

def CyclicGroupTriple(p):

    #defined Hopf algebras
    H = CyclicGroupAlgebra(p)

    #defines non-trivial pairing.
    R_elts = {}
    for i in range(p):
        for j in range(p):
            R_elts[(i,j)] = (1/p) * np.exp(-1 * 2.0j * (1/p) * np.pi * i * j)
    R_lb, R_sh = (0,1), (p,p)

    R = SparseTensor(R_elts, R_lb, R_sh)

    return QuasiTriangularTriple(H,R)

#Miscellaneous Hopf Algebras And Hopf Triples
    

#TRISECTION DIAGRAM CODE

#Trisection class

class TrisectionData:

    def __init__(self, g, k, intersect, curves, signs):

        self.g, self.k = g, k
        self.genus = self.g
        self.intersect = intersect

        self.curves = curves
        self.signs = signs

    def get_genus(self):

        return self.genus

    def get_intersection_number(self):

        return self.intersect

    def get_curves(self,i):

        if type(i) == int:
            return self.curves[i % 3]
        else:
            raise Exception("non-int index passed to TrisectionData.get_curves.")

    def sign(self,i):

        return self.signs[i]

    def __add__(self,T):

        S = self
        return connect_sum(S,T)

    def __sub__(self,T):

        S = self
        return connect_sum(S,orientation_reverse(T))

    def __mul__(self,m):

        if type(m) != int: 
            raise TypeError('invalid type ' + type(x) + ' passed to TrisectionData.__mul__')

        g, k, i, c, s = 0, 0, 0, [[],[],[]], ()
        T = TrisectionData(g,k,i,c,s)

        if m >= 0:
            S = self
        else:
            S = orientation_reverse(self)

        for i in range(abs(m)):
            T = T + S

        return T

    def __rmul__(self,m):

        return self * m

#Basic attribute functions

def genus(T):

    return T.get_genus()

def intersect(T):

    return T.get_intersection_number()

#Basic operations

def orientation_reverse(T):

    g, k, i, c = T.g, T.k, T.intersect, T.curves
    s = tuple([-1 * i for i in T.signs])

    return TrisectionData(g,k,i,c,s)

def connect_sum(S,T):

    g = S.g + T.g
    k = S.k + T.k
    i = S.intersect + T.intersect
    s = S.signs + T.signs

    C, sh = [], S.intersect
    for j in (0,1,2):
        C.append([])
        C[j] += [c for c in S.curves[j]]
        C[j] += [tuple([j + sh for j in c]) for c in T.curves[j]]

    return TrisectionData(g,k,i,C,s) 

def euler(T):

    return  2 + T.g - (3 * T.k)      

#Basic trisections

#Recall: k for trisection T for X is k = (2 + g(T) - euler_characteristic(X))/3

#stabilized S4.

g, k, i, c, s = 3, 1, 6, [[(0,1),(3,),(4,)],[(2,3),(5,),(0,)],[(4,5),(1,),(2,)]], (1,1,1,1,1,1)
S4 = TrisectionData(g,k,i,c,s)

#CP2.

g, k, i, c, s = 1, 0, 3, [[(0,2)],[(0,1)],[(1,2)]], (1,1,-1)
CP2 = TrisectionData(g,k,i,c,s)

#S2xS2.

g, k, i, c, s = 2, 0, 6, [[(0,1),(3,4)], [(0,2),(3,5)], [(1,5),(2,4)]], (-1,1,1,-1,1,1)
S2xS2 = TrisectionData(g,k,i,c,s)

#twS2xS2.

g, k, i, c, s = 2, 0, 6, [[(0,1),(3,4)], [(0,2),(3,5,6)], [(1,5),(2,6,4)]], (-1,1,1,-1,1,1,-1)
twS2xS2 = TrisectionData(g,k,i,c,s)

#S1xS3
g, k, i, c, s = 1, 1, 0, [[()],[()],[()]], ()
S1xS3 = TrisectionData(g,k,i,c,s)

#S2xT2
g, k, i, c, s = 4, 2, 24, [[(0,1),(2,3),(4,5),(6,7,8,9,10,11,12,13,14,15)],[(16,15),(13,17),(9,18),(1,19,2,20,10,21,5,22,8,23)],[(22,7),(11,21),(19,14),(0,17,12,4,6,16,3,20,18,23)]], (-1,1,-1,1,-1,1,-1,-1,1,-1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1)
S2xT2 = TrisectionData(g,k,i,c,s)

#S2xT2b
A = [(0,1,2,3,4,5), (6,7,8,9), (10,11,12,13,14,15,16,17,18,19), (20,21,22,23,24,25), (26,27,28,29,30,31), (32,33,34,35), (36,37,38,39)]
B = [(40,41,42,43,44,45), (46,47,48,49), (50,51,52,53,54,55,56,57,58,59), (20,60), (31,61), (32,62), (39,63)]
C = [(22,52,12,24,27,17,57,29), (26,18,58,44,4,38,33,1,41,51,11,25), (28,16,56,37,34,53,13,23), (10,50,42,2,47,7,0,40,21,60), (61,30,45,5,8,48,3,43,59,19), (62,46,6,14,54,35), (36,55,15,9,49,63)]
S = [1,1,-1,1,-1,-1,-1,-1,1,1,\
     1,-1,-1,-1,1,-1,1,1,1,-1,\
     1,-1,-1,1,1,1,-1,-1,-1,1,\
     1,-1,1,-1,-1,-1,1,1,1,-1,\
     -1,-1,1,-1,1,1,1,1,-1,-1,\
     -1,1,1,1,-1,1,-1,-1,-1,1,
     1,-1,-1,1]

g, k, i, c, s = 7, 3, 64, [A,B,C], S
S2xT2b = TrisectionData(g,k,i,c,s)

#INVARIANT CALCULATION CODE

#Computes trisection bracket.

def bracket(T, H, progress_report = False, sparcity_threshold = np.inf):

    #Computes trisection bracket.

    #Input:

    # T - a TrisectionData instance.
    # H - a Hopf triple instance.
    # progress_report - a Boolean governing whether or not certain progress measures are reported.
    # sparcity_threshold - an int determining the 

    #Output:

    # x - the Trisection-Kuperberg invariant #(T,H).

    #initiates curve-tensor list.
    #note that these are *NOT* organized by curve type,
    #i.e. 0/1/2 curves are all in the same same list.
    CT = []

    #for each curve type i and each i-type curve c, creates tensor
    #C -> D ->->-> corresponding to c and adds it to list.
    for i in [0,1,2]:

        #gets structure tensors of H[i] Hopf algebra for i-curves.
        C = H[i].get_cotrace()
        e = H[i].get_counit()
        D = H[i].get_coproduct()

        #computes maximum number of intersections on an i-curve.
        N = max([len(c) for c in T.get_curves(i)])

        #initiates list L of C -> D ->->-> cotrace to comultiplication tensors
        L = []

        #adds the 0-output entry C -> e and the 1-output entry C ->
        L.append(QuickContract(C,e,(0,),(0,)))
        L.append(C)

        #adds the 2-output to N+1-output entry.
        for j in range(2,N+1):
            Cj = QuickContract(L[j-1],D,(j-2,),(0,))
            L.append(Cj)

        #for each i-type curve...
        for c in T.get_curves(i):

            #computes the labels for the curve tensor.
            #labels are of form (i,j) where i is curve type and j is a intersection label.
            c_lb = tuple([(i,j) for j in c]) 

            #creates a curve tensor CTc for curve c.
            CTc = SparseTensor(L[len(c)].entries, c_lb, L[len(c)].shape)

            #appends curve to the curve tensor list CT[i].
            CT.append(CTc)

    #initates list of intersections.
    U = CT.pop()
    x = 1.0

    #initializes step counter (for progress reporting)
    step = 1

    #computes the fully contracted tensor U by step-by-step contractions
    #of U with new tensors in 
    while len(CT) != 0:

        #identifies curve tensor V with most connections to current U tensor.
        V, A_U, B_V = None, [], []
        for W in CT:

            A_U_new, B_W = [], []
            for i,j in U.labels:
                for k,l in W.labels:
                    if j == l:
                        A_U_new.append((i,j))
                        B_W.append((k,l))

            if len(B_W) > len(B_V):
                V, A_U, B_V = W, A_U_new, B_W

        #print('U labels: ' + str(U.labels))
        #print('V labels: ' + str(V.labels))
        #print('common labels: ' + str(A_U) + str(B_V))

        N = len(A_U)

        #progress report...
        if progress_report:
            print('----step: ' + str(step) + '----')
            print('log size of U: ' + str(log_tensor_size(U)))
            print('spcties (pairing phase):',end=' ')

        #if U has no indices, restarts process and replaces U with another
        #curve tensor U from CT. Saves value of old U in x.
        if len(U.labels) == 0:

            x = x * U.entries[()]
            U = CT.pop()

        #if U has outward connections, then V is non-trivial. contracts all
        #connections between U and V.
        else:

            for m in range(N):

                i,j = A_U[m]
                k,l = B_V[m]

                if T.sign(j) == -1:
                    Si = H[i].get_antipode().copy()
                    Si.relabel({0:(i,j),1:(i+3,j)})

                    #contracts antipodes with U outputs (uses labels).
                    U = QuickContract(U,Si,((i,j),),((i,j),),True)

                else:
                    U.relabel({(i,j):(i+3,j)})

                Pik = H(i,k).copy()
                Pik.relabel({i:(i+3,j),k:(k,j)})

                #contracts pairings with U outputs (uses index labels).
                U = QuickContract(U,Pik,((i+3,j),),((i+3,j),),True) 

                #prints progress and checks sparcity threshold.
                if progress_report: print(str(abs_sparcity(U)),end=' ')
                if abs_sparcity(U) > sparcity_threshold:
                    x = None
                    print('sparcity threshold exceeded. aborting calculation...')
                    break

            #contracts all matching outputs of V with those of U (uses labels)
            U = QuickContract(U,V,B_V,B_V,True) 
            
            #prints progress and checks sparcity threshold.
            if progress_report:
                print('')
                print('spcty (contraction phase): ' + str(abs_sparcity(U)))
                print(' ')
            if abs_sparcity(U) > sparcity_threshold:
                x = None
                print('sparcity threshold exceeded. aborting calculation...')
                break          

            #removes newly connected V from the list CT of un-contracted curve tensors.
            CT.remove(V)

        #increments step counter
        step += 1

    #print(U.labels)

    x = x * U.entries[()]

    return x

def bracket2(T, H, progress_report = False, sparcity_threshold = np.inf):

    #Computes trisection bracket.

    #Input:

    # T - a TrisectionData instance.
    # H - a Hopf triple instance.
    # progress_report - a Boolean governing whether or not certain progress measures are reported.
    # sparcity_threshold - an int determining  

    #Output:

    # x - the Trisection-Kuperberg invariant #(T,H).

    #initiates curve-tensor list.
    #note that these are *NOT* organized by curve type,
    #i.e. 0/1/2 curves are all in the same same list.
    CT = []

    #for each curve type i and each i-type curve c, creates tensor
    #C -> D ->->-> corresponding to c and adds it to list.
    for i in [0,1,2]:

        #gets structure tensors of H[i] Hopf algebra for i-curves.
        C = H[i].get_cotrace()
        e = H[i].get_counit()
        D = H[i].get_coproduct()

        #computes maximum number of intersections on an i-curve.
        N = max([len(c) for c in T.get_curves(i)])

        #initiates list L of C -> D ->->-> cotrace to comultiplication tensors
        L = []

        #adds the 0-output entry C -> e and the 1-output entry C ->
        L.append(QuickContract(C,e,(0,),(0,)))
        L.append(C)

        #adds the 2-output to N+1-output entry.
        for j in range(2,N+1):
            Cj = QuickContract(L[j-1],D,(j-2,),(0,))
            L.append(Cj)

        #for each i-type curve...
        for c in T.get_curves(i):

            #computes the labels for the curve tensor.
            #labels are of form (i,j) where i is curve type and j is a intersection label.
            c_lb = tuple([(i,j) for j in c]) 

            #creates a curve tensor CTc for curve c.
            CTc = SparseTensor(L[len(c)].entries, c_lb, L[len(c)].shape)

            #appends curve to the curve tensor list CT[i].
            CT.append(CTc)

    #initates list of intersections.
    U = CT.pop()
    x = 1.0

    #initializes step counter (for progress reporting)
    step = 1

    #computes the fully contracted tensor U by step-by-step contractions
    #of U with new tensors in 
    while len(CT) != 0:

        #identifies curve tensor V with most connections to current U tensor.
        V, A_U, B_V = None, [], []
        for W in CT:

            A_U_new, B_W = [], []
            for i,j in U.labels:
                for k,l in W.labels:
                    if j == l:
                        A_U_new.append((i,j))
                        B_W.append((k,l))

            if len(B_W) > len(B_V):
                V, A_U, B_V = W, A_U_new, B_W

        #print('U labels: ' + str(U.labels))
        #print('V labels: ' + str(V.labels))
        #print('common labels: ' + str(A_U) + str(B_V))

        N = len(A_U)

        #progress report...
        if progress_report:
            print('----step: ' + str(step) + '----')
            print('log size of U: ' + str(log_tensor_size(U)))
            print('spcties (pairing phase):',end=' ')

        #if U has no indices, restarts process and replaces U with another
        #curve tensor U from CT. Saves value of old U in x.
        if len(U.labels) == 0:

            x = x * U.entries[()]
            U = CT.pop()

        #if U has outward connections, then V is non-trivial. contracts all
        #connections between U and V.
        else:

            for m in range(N):

                i,j = A_U[m]
                k,l = B_V[m]

                if T.sign(j) == -1:
                    Si = H[i].get_antipode().copy()
                    Si.relabel({0:(i,j),1:(i+3,j)})

                    #contracts antipodes with U outputs (uses labels).
                    U = QuickContract(U,Si,((i,j),),((i,j),),True)

                else:
                    U.relabel({(i,j):(i+3,j)})

                Pik = H(i,k).copy()
                Pik.relabel({i:(i+3,j),k:(k,j)})

                #contracts pairings with U outputs (uses index labels).
                U = QuickContract(U,Pik,((i+3,j),),((i+3,j),),True) 

                #prints progress and checks sparcity threshold.
                if progress_report: print(str(abs_sparcity(U)),end=' ')

            #contracts all matching outputs of V with those of U (uses labels)
            U = QuickContract(U,V,B_V,B_V,True) 

            #removes newly connected V from the list CT of un-contracted curve tensors.
            CT.remove(V)

            spU = abs_sparcity(U)
            
            #prints progress and checks sparcity threshold.
            if progress_report:
                print('')
                print('spcty (contraction phase): ' + str(abs_sparcity(U)))
                print(' ')

            if spU > sparcity_threshold:

                if progress_report: print('exceeded sparcity threshold. swapping U.')

                iV, spV, N = U, spU, len(CT)
                for i in range(N):
                    W, spW = CT[i], abs_sparcity(CT[i])
                    if spW < spV:
                        iV, spV = i, spW

                CT.append(U)
                U = CT.pop(iV)

                if progress_report: print('changed to a tensor of sparcity: ', spV)

        #increments step counter
        step += 1

    #print(U.labels)

    x = x * U.entries[()]

    return x

#Computes trisection invariant.

def Kup(T,H):

    return bracket(T,H)/(bracket(S4,H) ** (genus(T)/3))

invariant = Kup

def Ksh(T,p):

    H = CyclicGroupTriple(p)
    return invariant(T,H) * (p ** (euler(T)/2 - 1))