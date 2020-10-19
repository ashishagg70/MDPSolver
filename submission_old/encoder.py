import argparse
import time
import numpy as np
from MDPInstance import MDPInstance

#########parser###############
ap = argparse.ArgumentParser()
ap.add_argument("--grid", required=True,
   help="first operand", default='./gridfile')

args = vars(ap.parse_args())
GridFilePath=args['grid']

grid = np.loadtxt(GridFilePath, dtype=int)
grid=np.array(grid)
#print(grid)
nr = len(grid)
nc = len(grid[0])
print("numStates ",nr*nc)
print("numActions 4")
start=np.where(grid.flatten()==2)[0][0]
end=np.where(grid.flatten()==3)[0][0]
print("start ",start)
print("end ", end)
'''
0 ->up, N
1 ->down, S
2 -> right, E
3 -> left, W
'''
for i in range(nr):
    for j in range(nc):
        if(grid[i][j]!=1 and grid[i][j]!=3):
            s=nc*i+j
            if(grid[i-1][j]==0 or grid[i-1][j]==2):
                s2=nc*(i-1)+j
                print("transition %d 0 %d -2 1"%(s,s2 ))
            elif (grid[i - 1][j] == 1):
                print("transition %d 0 %d -100000 1" % (s, s))
            elif (grid[i - 1][j] == 3):
                s2 = nc * (i - 1) + j
                print("transition %d 0 %d 1 1" % (s, s2))

            if (grid[i + 1][j] == 0 or grid[i+1][j]==2):
                s2 = nc * (i + 1) + j
                print("transition %d 1 %d -2 1" % (s, s2))
            elif (grid[i + 1][j] == 1):
                print("transition %d 1 %d -100000 1" % (s, s))
            elif (grid[i + 1][j] == 3):
                s2 = nc * (i + 1) + j
                print("transition %d 1 %d 1 1" % (s, s2))

            if (grid[i][j+1] == 0 or grid[i][j+1]==2):
                s2 = nc * i + (j+1)
                print("transition %d 2 %d -2 1" % (s, s2))
            elif (grid[i][j+1] == 1):
                print("transition %d 2 %d -100000 1" % (s, s))
            elif (grid[i][j + 1] == 3):
                s2 = nc * i + (j + 1)
                print("transition %d 2 %d 1 1" % (s, s2))

            if (grid[i][j-1] == 0 or grid[i][j-1]==2):
                s2 = nc * i + j-1
                print("transition %d 3 %d -2 1" % (s, s2))
            elif (grid[i][j-1] == 1):
                print("transition %d 3 %d -100000 1" % (s, s))
            elif (grid[i][j - 1] == 3):
                s2 = nc * i + j - 1
                print("transition %d 3 %d 1 1" % (s, s2))




print("mdptype episodic")
print("discount  0.96")

