import os
import numpy as np
import matplotlib.pyplot as plt
import random


class tetris:
    def __init__(self, n_rows, n_cols):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.screen_n_rows = n_rows
        self.screen_n_cols = n_cols
        self.possible_actions = (0, 1, 2)
        self.blockTypes = [ np.array(x,dtype=np.int32) for x in [[1,1],[1,0]], [[1,0],[1,1]],[[0,1],[1,1]], [[1,1],[0,1]], [[1,1],[0,0]], [[1,0],[1,0]]]
        self.reward_adrop = self.reward_epoch = 0

        # variables
        self.startNewEpoch()
    def getActionList(self):
        return self.possible_actions
    def getScreenSize(self):
        return (self.screen_n_rows, self.screen_n_cols)

    def update(self, action):
        """
        action:
            0: do nothing
            1: move left
            2: move right
        """
        self.terminal = False

        # Move left/right
        temp = np.zeros((self.screen_n_rows, self.screen_n_cols), dtype=np.int32)
        if   action == self.possible_actions[1]: temp[:, :-1] = self.block[:,1:  ]
        elif action == self.possible_actions[2]: temp[:,1:  ] = self.block[:, :-1]
        #elif action == self.possible_actions[3]: temp[:,1:  ] = self.block[:, :-1] # Rotate left
        #elif action == self.possible_actions[4]: temp[:,1:  ] = self.block[:, :-1] # Rotate right

        if temp.sum() == self.block.sum() and np.logical_and(temp,self.piles).sum()==0:
            self.block = temp

        # Move down
        temp = np.zeros((self.screen_n_rows, self.screen_n_cols),dtype=np.int32)
        temp[1:,:] = self.block[:-1,:]
        if temp.sum() == self.block.sum() and np.logical_and(temp,self.piles).sum()==0:
            self.block = temp
        else:
            self.piles += self.block
            self.block[:,:] = 0

        # Delete lines
        for i in range(self.screen_n_rows):
            if self.piles[i,:].sum() == self.screen_n_cols:
                self.reward_epoch += 1
                self.reward_adrop += 1
                self.piles[1:(i+1),:] = self.piles[:i,:]
                self.piles[0,:] = 0

        isStartNewDrop = False
        # loose check
        if self.piles[self.blockTypes[0].shape[0]-1,:].sum()>0:
            isStartNewDrop = True
            self.terminal = True

        # Check terminal
        if self.block.sum() == 0:
            isStartNewDrop = True

        # Reset if required
        if self.terminal:
            return False
        if isStartNewDrop:
            self.reset()
        return True

    def observe(self):
        screen  = np.zeros((self.screen_n_rows, self.screen_n_cols), dtype=np.int32)
        screen += self.piles
        screen += self.block
        if self.terminal:
            reward = -10
        else:
            reward  = self.reward_adrop
            self.reward_adrop = 0
        #return self.pile,self.block,screen, reward, self.terminal
        return screen, reward, self.terminal

    def execute_action(self, action):
        return self.update(action)

    def reset(self):
        self.block = np.zeros((self.screen_n_rows, self.screen_n_cols), dtype=np.int32)
        blockPos = np.random.randint(0,self.screen_n_cols-self.blockTypes[0].shape[1]+1)
        blockIdx = np.random.randint(len(self.blockTypes))
        for i in range(2):
            for j in range(2):
                self.block[i,j+blockPos] = self.blockTypes[blockIdx][i,j]
        self.terminal = False
        return

    def startNewEpoch(self):
        self.piles  = np.zeros((self.screen_n_rows, self.screen_n_cols), dtype=np.int32)
        self.reward_epoch = 0
        self.reset()
        return

if __name__=="__main__":
    t = tetris(20,10)
    fig = plt.figure(figsize=(10,5))
    fig.canvas.set_window_title("TeTris")
    epoch = 0
    while True:
        epoch += 1
        print("epoch=%d"%epoch)
        t.startNewEpoch()
        while t.execute_action(random.randint(0,2)):
            screen, _, _ = t.observe()
            img = plt.imshow(screen, interpolation="none", cmap="gray")
            fig.show()
            #plt.draw()
            plt.pause(1e-10)

