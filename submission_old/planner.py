import argparse
from MDPInstance import MDPInstance

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
if(algorithm=="vi"):
   value,policy=mdp.valueIteration(1e-8)
elif(algorithm=="hpi"):
   value,policy=mdp.harwardPolicyIteration()
elif(algorithm=="lp"):
   value,policy=mdp.linearProgramming()
for v,a in zip(value.tolist(), policy.tolist()):
   print(str(v) + " " + str(a))

