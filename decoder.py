import argparse
import numpy as np
from collections import defaultdict

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
stateToPosition=defaultdict()
sCount=0
for i in range(nr):
    for j in range(nc):
        if(grid[i][j]!=1 and grid[i][j]!=3):
            s=nc*i+j
            if s not in stateToPosition:
                stateToPosition[s]=sCount
                sCount+=1
            if(grid[i-1][j]==0 or grid[i-1][j]==2):
                s2=nc*(i-1)+j
                if s2 not in stateToPosition:
                    stateToPosition[s2] = sCount
                    sCount += 1
            elif (grid[i - 1][j] == 3):
                s2 = nc * (i - 1) + j
                if s2 not in stateToPosition:
                    stateToPosition[s2] = sCount
                    sCount += 1

            if (grid[i + 1][j] == 0 or grid[i+1][j]==2):
                s2 = nc * (i + 1) + j
                if s2 not in stateToPosition:
                    stateToPosition[s2] = sCount
                    sCount += 1
            elif (grid[i + 1][j] == 3):
                s2 = nc * (i + 1) + j
                if s2 not in stateToPosition:
                    stateToPosition[s2] = sCount
                    sCount += 1

            if (grid[i][j+1] == 0 or grid[i][j+1]==2):
                s2 = nc * i + (j+1)
                if s2 not in stateToPosition:
                    stateToPosition[s2] = sCount
                    sCount += 1
            elif (grid[i][j + 1] == 3):
                s2 = nc * i + (j + 1)
                if s2 not in stateToPosition:
                    stateToPosition[s2] = sCount
                    sCount += 1

            if (grid[i][j-1] == 0 or grid[i][j-1]==2):
                s2 = nc * i + j-1
                if s2 not in stateToPosition:
                    stateToPosition[s2] = sCount
                    sCount += 1
            elif (grid[i][j - 1] == 3):
                s2 = nc * i + j - 1
                if s2 not in stateToPosition:
                    stateToPosition[s2] = sCount
                    sCount += 1



curr=start
while(curr!=end):
    if(actionGrid1D[stateToPosition[curr]]==0):
        print('N',end=" ")
        curr-=nc
    elif(actionGrid1D[stateToPosition[curr]]==1):
        print('S',end=" ")
        curr+=nc
    elif (actionGrid1D[stateToPosition[curr]] == 2):
        print('E', end=" ")
        curr+=1
    elif (actionGrid1D[stateToPosition[curr]] == 3):
        print('W', end=" ")
        curr-=1
print()
#solGrid=np.array(policy).reshape((n,n))
