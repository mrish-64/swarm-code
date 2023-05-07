#import CostFun as fit
import random
import numpy
import math
import matplotlib.pyplot as plt   
import time
import sys
sys.path.append(r"D:\My_Code")
import functions

start_time=time.time()
lb=-10
ub=10
###################################################
dim=310
Max_iter=400
####################################################
SearchAgents_no=50

if not isinstance(lb, list):
   lb = [lb] * dim
if not isinstance(ub, list):
   ub = [ub] * dim
        
    
# initialize position vector and score for the leader
Leader_pos=numpy.zeros(dim)
Leader_score=float("inf")  #change this to -inf for maximization problems
    
    
#Initialize the positions of search agents
Positions = numpy.zeros((SearchAgents_no, dim))
for i in range(dim):
   Positions[:, i] = numpy.random.uniform(0,1,SearchAgents_no) *(ub[i]-lb[i])+lb[i]
    
#Initialize convergence
convergence_curve=numpy.zeros(Max_iter)
    
t=0  # Loop counter
   
# Main loop
while t<Max_iter:
  for i in range(0,SearchAgents_no):
            
    # Return back the search agents that go beyond the boundaries of the search space
            
    #Positions[i,:]=checkBounds(Positions[i,:],lb,ub)
     for j in range(dim):        
       Positions[i,j]=numpy.clip(Positions[i,j], lb[j], ub[j])
          
       # Calculate objective function for each search agent
       #fitness=fit.F1(Positions[i,:])
       #Convert to binary
       sigV=1/(1+(numpy.exp((-Positions[i,:]))))
       temp_position =( sigV > numpy.random.rand(Positions[i,:].size)).astype(int)
       fitness=functions.f(temp_position)
       Positions[i,:]=temp_position     
       # Update the leader
       if fitness<Leader_score: # Change this to > for maximization problem
          Leader_score=fitness; # Update alpha
          Leader_pos=Positions[i,:].copy() # copy current whale position into the leader position
            
            
        
        
  a=2-t*((2)/Max_iter); # a decreases linearly fron 2 to 0 in Eq. (2.3)
        
        # a2 linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
  a2=-1+t*((-1)/Max_iter);
  # Update the Position of search agents
  for i in range(0,SearchAgents_no):
      r1=random.random() # r1 is a random number in [0,1]
      r2=random.random() # r2 is a random number in [0,1]
            
      A=2*a*r1-a  # Eq. (2.3) in the cited paper
      C=2*r2      # Eq. (2.4) in the cited paper
            
            
      b=1;               #  parameters in Eq. (2.5)
      l=(a2-1)*random.random()+1   #  parameters in Eq. (2.5)
            
      p = random.random()        # p in Eq. (2.6)
            
      for j in range(0,dim):
                
          if p<0.5:
             if abs(A)>=1:
                rand_leader_index = math.floor(SearchAgents_no*random.random());
                X_rand = Positions[rand_leader_index, :]
                D_X_rand=abs(C*X_rand[j]-Positions[i,j]) 
                Positions[i,j]=X_rand[j]-A*D_X_rand      
                        
             elif abs(A)<1:
                D_Leader=abs(C*Leader_pos[j]-Positions[i,j]) 
                Positions[i,j]=Leader_pos[j]-A*D_Leader      
                    
          elif p>=0.5:
             distance2Leader=abs(Leader_pos[j]-Positions[i,j])
                    # Eq. (2.5)
             Positions[i,j]=distance2Leader*math.exp(b*l)*math.cos(l*2*math.pi)+Leader_pos[j]
                    
      
        
  convergence_curve[t]=Leader_score
  if (t%1==0):
      print(['At iteration '+ str(t)+ ' the best fitness is '+ str(Leader_score)]);
  t=t+1



# Data for plotting
plt.plot(convergence_curve)
plt.xlabel('Iteration') 
plt.ylabel('Cost') 
plt.title('Cost Function in Itterations') 

end_time=time.time()
print(f" WOA takes {end_time-start_time} second")  
# function to show the plot 
plt.show()  
#
print(Leader_pos)
print(functions.f([int(x) for x in Leader_pos]))
ans=list(dict.fromkeys(convergence_curve))
print(ans)



