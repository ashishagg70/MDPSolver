import argparse
import time
import numpy as np
from MDPInstance import MDPInstance

#########parser###############
ap = argparse.ArgumentParser()
ap.add_argument("--grid", required=True,
   help="first operand", default='./gridfile')
ap.add_argument("--value_policy", required=True,
   help="second operand", default='./value_and_policy_file')

args = vars(ap.parse_args())
GridFilePath=args['grid']
ValuePolicyFilePath=args['value_policy']

valuePolicy = np.loadtxt(ValuePolicyFilePath, dtype=float)
grid = np.loadtxt(GridFilePath, dtype=int)
nr = len(grid)
nc = len(grid[0])
start=np.where(grid.flatten()==2)[0][0]
end=np.where(grid.flatten()==3)[0][0]
actionGrid1D = np.array(valuePolicy[:,1]).astype(int)
curr=start
while(curr!=end):
    if(actionGrid1D[curr]==0):
        print('N',end=" ")
        curr-=nc
    elif(actionGrid1D[curr]==1):
        print('S',end=" ")
        curr+=nc
    elif (actionGrid1D[curr] == 2):
        print('E', end=" ")
        curr+=1
    elif (actionGrid1D[curr] == 3):
        print('W', end=" ")
        curr-=1
print()
