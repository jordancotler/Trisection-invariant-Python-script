import tri

def Algebra():

    #constructs coproduct D

    D = tri.SparseTensor({}, (0,1,2),(8,8,8))
    
    D[(0,0,0)] = 1.0
    D[(1,1,1)] = 1.0
    D[(2,2,2)] = 1.0
    D[(3,3,3)] = 1.0

    D[(4,4,4)] = .5
    D[(4,6,4)] = .5
    D[(4,4,5)] = .5
    D[(4,6,5)] = -.5

    D[(5,5,5)] = .5
    D[(5,7,5)] = .5
    D[(5,5,4)] = .5
    D[(5,7,4)] = -.5

    D[(6,6,6)] = .5
    D[(6,4,6)] = .5
    D[(6,6,7)] = .5
    D[(6,4,7)] = -.5

    D[(7,7,7)] = .5
    D[(7,5,7)] = .5
    D[(7,7,6)] = .5
    D[(7,5,6)] = -.5

    #constructs counit e

    e = tri.SparseTensor({}, (0,),(8,))

    for i in range(8): e[(i,)] = 1

    #constructs antipode S

    S = tri.SparseTensor({}, (0,1),(8,8))

    S[(0,0)] = 1.0
    S[(1,1)] = 1.0
    S[(2,2)] = 1.0
    S[(3,3)] = 1.0
    S[(4,4)] = 1.0
    S[(5,6)] = 1.0
    S[(6,5)] = 1.0
    S[(7,7)] = 1.0

    return tri.PartHopfAlgebra(D,e,S,8)

def P0():

    P = tri.SparseTensor({},(0,1),(8,8))
    sq2 = 2 ** .5

    P[(0,0)], P[(0,1)], P[(0,2)], P[(0,3)], P[(0,4)], P[(0,5)], P[(0,6)], P[(0,7)] = 1.0,1.0,1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    P[(1,0)], P[(1,1)], P[(1,2)], P[(1,3)], P[(1,4)], P[(1,5)], P[(1,6)], P[(1,7)] = 1.0,-1.0,-1.0, 1.0, 1.0j, -1.0j, -1.0j, 1.0j
    P[(2,0)], P[(2,1)], P[(2,2)], P[(2,3)], P[(2,4)], P[(2,5)], P[(2,6)], P[(2,7)] = 1.0,-1.0,-1.0, 1.0, 1.0j, -1.0j, -1.0j, 1.0j
    P[(3,0)], P[(3,1)], P[(3,2)], P[(3,3)], P[(3,4)], P[(3,5)], P[(3,6)], P[(3,7)] = 1.0,1.0,1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    P[(4,0)], P[(4,1)], P[(4,2)], P[(4,3)], P[(4,4)], P[(4,5)], P[(4,6)], P[(4,7)] = 1.0, 1.0j, 1.0j, 1.0, -1 - 1.0j, 0.0, 0.0, -1 - 1.0j
    P[(5,0)], P[(5,1)], P[(5,2)], P[(5,3)], P[(5,4)], P[(5,5)], P[(5,6)], P[(5,7)] = 1.0, -1.0j, -1.0j, 1.0, 0.0, -1.0 + 1.0j, -1.0 + 1.0j, 0.0
    P[(6,0)], P[(6,1)], P[(6,2)], P[(6,3)], P[(6,4)], P[(6,5)], P[(6,6)], P[(6,7)] = 1.0, -1.0j, -1.0j, 1.0, 0.0, -1.0 + 1.0j, -1.0 + 1.0j, 0.0
    P[(7,0)], P[(7,1)], P[(7,2)], P[(7,3)], P[(7,4)], P[(7,5)], P[(7,6)], P[(7,7)] = 1.0, 1.0j, 1.0j, 1.0, -1.0 - 1.0j, 0.0, 0.0, -1.0 - 1.0j

    return P

def P1():

    P = tri.SparseTensor({},(0,1),(8,8))
    sq2 = 2 ** .5

    P[(0,0)], P[(0,1)], P[(0,2)], P[(0,3)], P[(0,4)], P[(0,5)], P[(0,6)], P[(0,7)] = 1.0,1.0,1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    P[(1,0)], P[(1,1)], P[(1,2)], P[(1,3)], P[(1,4)], P[(1,5)], P[(1,6)], P[(1,7)] = 1.0,-1.0,-1.0, 1.0, 1.0j, -1.0j, -1.0j, 1.0j
    P[(2,0)], P[(2,1)], P[(2,2)], P[(2,3)], P[(2,4)], P[(2,5)], P[(2,6)], P[(2,7)] = 1.0,-1.0,-1.0, 1.0, -1.0j, 1.0j, 1.0j, -1.0j
    P[(3,0)], P[(3,1)], P[(3,2)], P[(3,3)], P[(3,4)], P[(3,5)], P[(3,6)], P[(3,7)] = 1.0,1.0,1.0, 1.0, -1.0, -1.0, -1.0, -1.0
    P[(4,0)], P[(4,1)], P[(4,2)], P[(4,3)], P[(4,4)], P[(4,5)], P[(4,6)], P[(4,7)] = 1.0, -1.0j, 1.0j, -1.0, -1*sq2, 0.0, 0.0, 1*sq2
    P[(5,0)], P[(5,1)], P[(5,2)], P[(5,3)], P[(5,4)], P[(5,5)], P[(5,6)], P[(5,7)] = 1.0, 1.0j, -1.0j, -1.0, 0.0, 1.0j * sq2, -1.0j*sq2, 0.0
    P[(6,0)], P[(6,1)], P[(6,2)], P[(6,3)], P[(6,4)], P[(6,5)], P[(6,6)], P[(6,7)] = 1.0, 1.0j, -1.0j, -1.0, 0.0, -1.0j * sq2, 1.0j * sq2, 0.0
    P[(7,0)], P[(7,1)], P[(7,2)], P[(7,3)], P[(7,4)], P[(7,5)], P[(7,6)], P[(7,7)] = 1.0, -1.0j, 1.0j, -1.0, sq2, 0.0, 0.0, -1*sq2

    return P 

def P2():

    P = tri.SparseTensor({},(0,1),(8,8))
    sq2 = 2 ** .5

    P[(0,0)], P[(0,1)], P[(0,2)], P[(0,3)], P[(0,4)], P[(0,5)], P[(0,6)], P[(0,7)] = 1.0,1.0,1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    P[(1,0)], P[(1,1)], P[(1,2)], P[(1,3)], P[(1,4)], P[(1,5)], P[(1,6)], P[(1,7)] = 1.0,-1.0,-1.0, 1.0, 1.0j, -1.0j, -1.0j, 1.0j
    P[(2,0)], P[(2,1)], P[(2,2)], P[(2,3)], P[(2,4)], P[(2,5)], P[(2,6)], P[(2,7)] = 1.0,-1.0,-1.0, 1.0, -1.0j, 1.0j, 1.0j, -1.0j
    P[(3,0)], P[(3,1)], P[(3,2)], P[(3,3)], P[(3,4)], P[(3,5)], P[(3,6)], P[(3,7)] = 1.0,1.0,1.0, 1.0, -1.0, -1.0, -1.0, -1.0
    P[(4,0)], P[(4,1)], P[(4,2)], P[(4,3)], P[(4,4)], P[(4,5)], P[(4,6)], P[(4,7)] = 1.0, -1.0j, 1.0j, -1.0, sq2, 0.0, 0.0, -1*sq2
    P[(5,0)], P[(5,1)], P[(5,2)], P[(5,3)], P[(5,4)], P[(5,5)], P[(5,6)], P[(5,7)] = 1.0, 1.0j, -1.0j, -1.0, 0.0, -1.0j * sq2, 1.0j*sq2, 0.0
    P[(6,0)], P[(6,1)], P[(6,2)], P[(6,3)], P[(6,4)], P[(6,5)], P[(6,6)], P[(6,7)] = 1.0, 1.0j, -1.0j, -1.0, 0.0, 1.0j * sq2, -1.0j * sq2, 0.0
    P[(7,0)], P[(7,1)], P[(7,2)], P[(7,3)], P[(7,4)], P[(7,5)], P[(7,6)], P[(7,7)] = 1.0, -1.0j, 1.0j, -1.0, -1*sq2, 0.0, 0.0, sq2

    return P

def P3():

    P = tri.SparseTensor({},(0,1),(8,8))
    sq2 = 2 ** .5

    P[(0,0)], P[(0,1)], P[(0,2)], P[(0,3)], P[(0,4)], P[(0,5)], P[(0,6)], P[(0,7)] = 1.0,1.0,1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    P[(1,0)], P[(1,1)], P[(1,2)], P[(1,3)], P[(1,4)], P[(1,5)], P[(1,6)], P[(1,7)] = 1.0,-1.0,-1.0, 1.0, -1.0j, 1.0j, 1.0j, -1.0j
    P[(2,0)], P[(2,1)], P[(2,2)], P[(2,3)], P[(2,4)], P[(2,5)], P[(2,6)], P[(2,7)] = 1.0,-1.0,-1.0, 1.0, 1.0j, -1.0j, -1.0j, 1.0j
    P[(3,0)], P[(3,1)], P[(3,2)], P[(3,3)], P[(3,4)], P[(3,5)], P[(3,6)], P[(3,7)] = 1.0,1.0,1.0, 1.0, -1.0, -1.0, -1.0, -1.0
    P[(4,0)], P[(4,1)], P[(4,2)], P[(4,3)], P[(4,4)], P[(4,5)], P[(4,6)], P[(4,7)] = 1.0, 1.0j, -1.0j, -1.0, -1*sq2, 0.0, 0.0, sq2
    P[(5,0)], P[(5,1)], P[(5,2)], P[(5,3)], P[(5,4)], P[(5,5)], P[(5,6)], P[(5,7)] = 1.0, -1.0j, 1.0j, -1.0, 0.0, -1.0j * sq2, 1.0j*sq2, 0.0
    P[(6,0)], P[(6,1)], P[(6,2)], P[(6,3)], P[(6,4)], P[(6,5)], P[(6,6)], P[(6,7)] = 1.0, -1.0j, 1.0j, -1.0, 0.0, 1.0j * sq2, -1.0j * sq2, 0.0
    P[(7,0)], P[(7,1)], P[(7,2)], P[(7,3)], P[(7,4)], P[(7,5)], P[(7,6)], P[(7,7)] = 1.0, 1.0j, -1.0j, -1.0, sq2, 0.0, 0.0, -1*sq2

    return P


def TripletA():

	H0, H1, H2 = Algebra(), Algebra(), Algebra()

	P01, P12, P20 = P1(), P1(), P1()

	return tri.HopfTriple([H0,H1,H2],[P01,P12,P20])

def TripletB():

	H0, H1, H2 = Algebra(), Algebra(), Algebra()

	P01, P12, P20 = P1(), P2(), P3()

	return tri.HopfTriple([H0,H1,H2],[P01,P12,P20])


def TripletC():

	H0, H1, H2 = Algebra(), Algebra(), Algebra()

	P01, P12, P20 = P0(), P0(), P1()

	return tri.HopfTriple([H0,H1,H2],[P01,P12,P20])