import math
import numpy
import random
import time
# import benchmarks as Fit    
import matplotlib.pyplot as plt 
import sys
sys.path.append(r"D:\My_Code")
import functions 


    
def get_cuckoos(nest,best,lb,ub,n,dim):
    
    # perform Levy flights
    tempnest=numpy.zeros((n,dim))
    tempnest=numpy.array(nest)
    beta=3/2
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta);

    s=numpy.zeros(dim)
    for j in range (0,n):
        s=nest[j,:]
        u=numpy.random.randn(len(s))*sigma
        v=numpy.random.randn(len(s))
        step=u/abs(v)**(1/beta)
 
        stepsize=0.01*(step*(s-best))

        s=s+stepsize*numpy.random.randn(len(s))
    
        for k in range(dim):
            tempnest[j,k]=numpy.clip(s[k], lb[k], ub[k])

    return tempnest

def get_best_nest(nest,newnest,fitness,n,dim):
# Evaluating all new solutions
    tempnest=numpy.zeros((n,dim))
    tempnest=numpy.copy(nest)

    for j in range(0,n):
    #for j=1:size(nest,1),
         #convert position to binary
        sigV=1/(1+(numpy.exp((-newnest[j,:]))))
        temp_position =( sigV > numpy.random.rand(newnest[j,:].size)).astype(int)
        newnest[j,:]=temp_position
        fnew=functions.f(newnest[j,:].astype(int))
        if fnew<=fitness[j]:
           fitness[j]=fnew
           tempnest[j,:]=newnest[j,:]
        
    # Find the current best

    fmin = min(fitness)
    K=numpy.argmin(fitness)
    bestlocal=tempnest[K,:]

    return fmin,bestlocal,tempnest,fitness

# Replace some nests by constructing new solutions/nests
def empty_nests(nest,pa,n,dim):

    # Discovered or not 
    tempnest=numpy.zeros((n,dim))

    K=numpy.random.uniform(0,1,(n,dim))>pa
    
    
    stepsize=random.random()*(nest[numpy.random.permutation(n),:]-nest[numpy.random.permutation(n),:])

    
    tempnest=nest+stepsize*K
 
    return tempnest
##########################################################################


start_time=time.time()

lb=-10
ub=10
#################################################################################
dim=310;
N_IterTotal=1000
#####################################################################################
n=50
#N_IterTotal=1000

# Discovery rate of alien eggs/solutions
pa=0.25
    
    
nd=dim
    
    
#    Lb=[lb]*nd
#    Ub=[ub]*nd
convergence=[]
if not isinstance(lb, list):
    lb = [lb] * dim
if not isinstance(ub, list):
     ub = [ub] * dim

# RInitialize nests randomely
nest = numpy.zeros((n, dim))
for i in range(dim):
  nest[:, i] = numpy.random.uniform(0,1, n) * (ub[i] - lb[i]) + lb[i]
       
    
new_nest=numpy.zeros((n,dim))
new_nest=numpy.copy(nest)
    
bestnest=[0]*dim;
     
fitness=numpy.zeros(n) 
fitness.fill(float("inf"))
    
fmin,bestnest,nest,fitness =get_best_nest(nest,new_nest,fitness,n,dim)
convergence = [];
# Main loop counter
for iter in range (0,N_IterTotal):
# Generate new solutions (but keep the current best)
     
   new_nest=get_cuckoos(nest,bestnest,lb,ub,n,dim)
         
   # Evaluate new solutions and find best
   fnew,best,nest,fitness=get_best_nest(nest,new_nest,fitness,n,dim)
   new_nest=empty_nests(new_nest,pa,n,dim) ;
         
   # Evaluate new solutions and find best
   fnew,best,nest,fitness=get_best_nest(nest,new_nest,fitness,n,dim)
 
   if fnew<fmin:
      fmin=fnew
      bestnest=best
    
   
   print(['At iteration '+ str(iter)+ ' the best fitness is '+ str(fmin)]);
   convergence.append(fmin)

# Data for plotting
plt.plot(convergence)
plt.xlabel('Iteration') 
plt.ylabel('Cost') 
plt.title('Cost Function in Itterations') 
end_time=time.time()
print(f"CS takes {end_time-start_time} second")  
# function to show the plot 
plt.show() 
print(bestnest)   
print(convergence[-1])
print(functions.f([int(x) for x in bestnest])) 

ans=list(dict.fromkeys(convergence))
print(ans)




