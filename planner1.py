import argparse
import time
from MDPInstance import MDPInstance
import numpy as np

#########parser###############
ap = argparse.ArgumentParser()
ap.add_argument("--mdp", required=True,
   help="first operand", default='./data/mdp/continuing-mdp-2-2.txt')

ap.add_argument("--algorithm", required=True,
   help="second operand", default='vi')
args = vars(ap.parse_args())
MDPFilePath=args['mdp']
algorithm=args['algorithm']
#########END parser###############

mdp=MDPInstance(MDPFilePath)
#print("transtion=",mdp.transition[0][0])
#print("reward", mdp.reward[0][0])
#print("reward*transition", mdp.reward[0][0].dot(mdp.transition[0][0]))
#print(mdp.transition[9][4][0], " ", mdp.reward[9][4][0]," ",mdp.end)
#print(mdp.rawTransition)
if(algorithm=="vi"):
   start_time = time.time()
   value,policy=mdp.valueIteration(1e-8)
   print("--- %s seconds ---" % (time.time() - start_time))
elif(algorithm=="hpi"):
   start_time = time.time()
   value,policy=mdp.harwardPolicyIteration()
   print("--- %s seconds ---" % (time.time() - start_time))
elif(algorithm=="lp"):
   start_time = time.time()
   value,policy=mdp.linearProgramming()
   print("--- %s seconds ---" % (time.time() - start_time))
n=int(np.sqrt(mdp.numStates))
print(np.array(policy).reshape((n,n)))
#print("value=", value, "policy=",policy)
#for v,a in zip(value.tolist(), policy.tolist()):
#   print(str(v) + " " + str(a))

