import random
import numpy
import math
import time
# import TestFun as fit
import matplotlib.pyplot as plt
import sys
sys.path.append(r"D:\My_Code")
import functions


def Levy(dim):
    beta = 1.5
    sigma = (math.gamma(1+beta)*math.sin(math.pi*beta/2) /
             (math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = 0.01*numpy.random.randn(dim)*sigma
    v = numpy.random.randn(dim)
    zz = numpy.power(numpy.absolute(v), (1/beta))
    step = numpy.divide(u, zz)
    return step


dim = 310
SearchAgents_no = 50
lb = -100
ub = 100
Max_iter = 1000
# Fobj=fit.F6;
start_time=time.time()

# initialize the location and Energy of the rabbit
Rabbit_Location = numpy.zeros(dim)
Rabbit_Energy = float("inf")  # change this to -inf for maximization problems

if not isinstance(lb, list):
    lb = [lb for _ in range(dim)]
    ub = [ub for _ in range(dim)]
lb = numpy.asarray(lb)
ub = numpy.asarray(ub)

# Initialize the locations of Harris' hawks
X = numpy.asarray(
    [x*(ub-lb)+lb for x in numpy.random.uniform(0, 1, (SearchAgents_no, dim))])

# Initialize convergence
convergence_curve = numpy.zeros(Max_iter)

t = 0  # Loop counter

# Main loop
while t < Max_iter:
    for i in range(0, SearchAgents_no):

        # Check boundries

        X[i, :] = numpy.clip(X[i, :], lb, ub)
        sigV=1/(1+(numpy.exp((-X[i,:]))))
        temp_position =( sigV > numpy.random.rand(X[i,:].size))
        X[i,:]=temp_position
        # fitness of locations
        fitness = functions.f(X[i, :].astype(int))

        # Update the location of Rabbit
        if fitness < Rabbit_Energy:  # Change this to > for maximization problem
            Rabbit_Energy = fitness
            Rabbit_Location = X[i, :].copy()

    E1 = 2*(1-(t/Max_iter))  # factor to show the decreaing energy of rabbit

    # Update the location of Harris' hawks
    for i in range(0, SearchAgents_no):

        E0 = 2*random.random()-1  # -1<E0<1
        # escaping energy of rabbit Eq. (3) in the cited paper
        Escaping_Energy = E1*(E0)

        # -------- Exploration phase Eq. (1) in cited paper -------------------

        if abs(Escaping_Energy) >= 1:
         # Harris' hawks perch randomly based on 2 strategy:
            q = random.random()
            rand_Hawk_index = math.floor(SearchAgents_no*random.random())
            X_rand = X[rand_Hawk_index, :]
            if q < 0.5:
              # perch based on other family members
                X[i, :] = X_rand-random.random()*abs(X_rand-2 *
                                                     random.random()*X[i, :])
            elif q >= 0.5:
              # perch on a random tall tree (random site inside group's home range)
                X[i, :] = (Rabbit_Location - X.mean(0)) - \
                    random.random()*((ub-lb)*random.random()+lb)

               # -------- Exploitation phase -------------------
        elif abs(Escaping_Energy) < 1:
            # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

            # phase 1: ----- surprise pounce (seven kills) ----------
            # surprise pounce (seven kills): multiple, short rapid dives by different hawks

            r = random.random()  # probablity of each event

            # Hard besiege Eq. (6) in cited paper
            if r >= 0.5 and abs(Escaping_Energy) < 0.5:
                X[i, :] = (Rabbit_Location)-Escaping_Energy * \
                    abs(Rabbit_Location-X[i, :])

            # Soft besiege Eq. (4) in cited paper
            if r >= 0.5 and abs(Escaping_Energy) >= 0.5:
                # random jump strength of the rabbit
                Jump_strength = 2*(1 - random.random())
                X[i, :] = (Rabbit_Location-X[i, :])-Escaping_Energy * \
                    abs(Jump_strength*Rabbit_Location-X[i, :])

            # phase 2: --------performing team rapid dives (leapfrog movements)----------

            # Soft besiege Eq. (10) in cited paper
            if r < 0.5 and abs(Escaping_Energy) >= 0.5:
                # rabbit try to escape by many zigzag deceptive motions
                Jump_strength = 2*(1-random.random())
                X1 = Rabbit_Location-Escaping_Energy * \
                    abs(Jump_strength*Rabbit_Location-X[i, :])
                X1 = numpy.clip(X1, lb, ub)
                #convert X1 to binary ###################################################
                sigV=1/(1+(numpy.exp((-X1))))
                temp_position =( sigV > numpy.random.rand(X1.size))
                X1=temp_position

                if functions.f(X1.astype(int)) < fitness:  # improved move?
                    X[i, :] = X1.copy()
                else:  # hawks perform levy-based short rapid dives around the rabbit
                    X2 = Rabbit_Location-Escaping_Energy * \
                        abs(Jump_strength*Rabbit_Location -
                            X[i, :])+numpy.multiply(numpy.random.randn(dim), Levy(dim))
                    X2 = numpy.clip(X2, lb, ub)
                    #convert X2 to binary ###################################################
                    sigV=1/(1+(numpy.exp((-X2))))
                    temp_position =( sigV > numpy.random.rand(X2.size))                    
                    X2=temp_position
                    if functions.f(X2.astype(int)) < fitness:
                        X[i, :] = X2.copy()
            # Hard besiege Eq. (11) in cited paper
            if r < 0.5 and abs(Escaping_Energy) < 0.5:
                Jump_strength = 2*(1-random.random())
                X1 = Rabbit_Location-Escaping_Energy * \
                    abs(Jump_strength*Rabbit_Location-X.mean(0))
                X1 = numpy.clip(X1, lb, ub)
               #convert X1 to binary ###################################################
                sigV=1/(1+(numpy.exp((-X1))))
                temp_position =( sigV > numpy.random.rand(X1.size))                
                X1=temp_position
                if functions.f(X1.astype(int)) < fitness:  # improved move?
                    X[i, :] = X1.copy()
                else:  # Perform levy-based short rapid dives around the rabbit
                    X2 = Rabbit_Location-Escaping_Energy * \
                        abs(Jump_strength*Rabbit_Location-X.mean(0)) + \
                        numpy.multiply(numpy.random.randn(dim), Levy(dim))
                    X2 = numpy.clip(X2, lb, ub)
                    #convert X2 to binary ###################################################
                    sigV=1/(1+(numpy.exp((-X2))))
                    temp_position =( sigV > numpy.random.rand(X2.size))                    
                    X2=temp_position
                    if functions.f(X2.astype(int)) < fitness:
                        X[i, :] = X2.copy()

    convergence_curve[t] = Rabbit_Energy
    if (t % 1 == 0):
        print(['At iteration ' + str(t) +
              ' the best fitness is ' + str(Rabbit_Energy)])
    t = t+1

# Data for plotting
plt.semilogy(convergence_curve)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function in Itterations')
end_time=time.time()
print(f"HHO takes {end_time-start_time} second")
# function to show the plot
plt.show()
ans=list(dict.fromkeys(convergence_curve))
print(ans)
