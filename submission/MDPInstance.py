import numpy as np
import pulp as pulp

class MDPInstance:
    def __init__(self, filePath):
        self.filePath=filePath
        self.numStates=0
        self.numActions=0
        self.start=0
        self.end=0
        self.transition=[]
        #self.rawTransition = list()
        self.reward=[]
        self.action=[]
        self.mdptype=''
        self.discount=0
        self.parse()
    def parse(self):
        with open(self.filePath) as mdpFile:
            for line in mdpFile:
                word = [x for x in line.split()]
                if (word[0] == 'numStates'):
                    self.numStates=int(word[1])
                elif (word[0] == 'numActions'):
                    self.numActions=int(word[1])
                    self.transition = np.zeros((self.numStates,self.numActions, self.numStates), dtype=float)
                    self.reward = np.zeros((self.numStates,self.numActions,self.numStates), dtype=float)
                    #self.rawTransition = [list() for x in range(self.numStates)]
                elif (word[0] == 'start'):
                    self.start=int(word[1])
                elif (word[0] == 'end'):
                    self.end=[int(x) for x in word[1:]]
                elif (word[0] == 'mdptype'):
                    self.mdptype=word[1]
                elif (word[0] == 'discount'):
                    self.discount=float(word[1])
                elif (word[0] == 'transition'):
                    self.reward[int(word[1])][int(word[2])][int(word[3])] = float(word[4])
                    self.transition[int(word[1])][int(word[2])][int(word[3])]=float(word[5])
                    #self.rawTransition[int(word[1])].append(( int(word[2]), int(word[3]), float(word[4]), float(word[5])))
    def valueIteration(self, epsilon):
        numActions, numStates, y, T, R=self.numActions, self.numStates, self.discount, self.transition, self.reward
        V=np.zeros(numStates, dtype=float)
        pi=np.zeros(numStates, dtype=int)
        while True:
            Vold=np.copy(V)

            #* ijk,ijk ->ij to make fully vectorized
            for s in range(numStates):
                V[s]=np.max(np.einsum('ij,ij->i',T[s],R[s]+y*Vold))
            if(np.all(np.abs(V-Vold)<epsilon)):
                break
        V=np.round(V, decimals=6)

        for s in range(numStates):
            pi[s]=np.argmax(np.einsum('ij,ij->i',T[s],R[s]+y*Vold))

        return (V, pi)

    def harwardPolicyIteration(self):
        numActions, numStates, y, T, R = self.numActions, self.numStates, self.discount, self.transition, self.reward
        pi = np.zeros(numStates, dtype=int)
        while True:
            V=self.policyEvaluation(pi, T, R, numStates, y)
            pi_old = np.copy(pi)
            for s in range(numStates):
                pi[s]= np.argmax(np.einsum('ij,ij->i',T[s],R[s]+y*V))
            if(np.all(pi==pi_old)):
                break

        return (np.round(V, decimals=6), pi)

    def policyEvaluation(self, pi, T, R, numStates, y):
        #print("in policy eval")
        T2 = np.array(T[0][pi[0]][:])
        R2 = np.array(R[0][pi[0]][:])
        for s in range(1, numStates):
            T2 = np.vstack((T2, T[s][pi[s]][:]))
            R2 = np.vstack((R2, R[s][pi[s]][:]))
        A = (np.eye(numStates) - y * T2)
        b = np.einsum('ij,ij->i', T2, R2)
        V = np.linalg.solve(A, b)
        #print("out policy eval")
        return V

    def linearProgramming(self):
        numActions, numStates, y, T, R = self.numActions, self.numStates, self.discount, self.transition, self.reward
        pi = np.zeros(numStates, dtype=int)
        V = np.zeros(numStates, dtype=float)
        Lp_prob = pulp.LpProblem('policyEvaluation', pulp.LpMinimize)

        #adding var
        ValueVar = []
        for i in range(numStates):
            variable = str('V' + str(i))
            variable = pulp.LpVariable(variable)
            ValueVar.append(variable)
        ValueVar=np.array(ValueVar)
        #Objective
        Lp_prob += ValueVar.sum()
        # Constraints:
        '''for s,listItem in enumerate(transitionList):
            print([ p*(r+y*ValueVar[s2]) for a, s2, r, p in listItem ])
            Lp_prob +=ValueVar[s]>=pulp.lpSum([ p*(r+y*ValueVar[s2]) for a, s2, r, p in transitionList[s] ])'''
        for s in range(numStates):
            for a in range(numActions):
                Lp_prob += ValueVar[s]>=  T[s][a].dot((R[s][a]+y*ValueVar))
        status = pulp.PULP_CBC_CMD(msg=0).solve(Lp_prob)
        #status = pulp.PULP_CBC_CMD(msg=0,path='/Library/Python/3.8/site-packages/pulp/solverdir/cbc/linux/64/cbc').solve(Lp_prob)
        assert status == pulp.LpStatusOptimal
        for i, v in enumerate(ValueVar):
            V[i]=pulp.value(v)
        V = np.round(V, decimals=6)
        for s in range(numStates):
            pi[s] = np.argmax(np.einsum('ij,ij->i', T[s], R[s] + y * V))
        return (V, pi)
