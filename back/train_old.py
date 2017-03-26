import numpy as np
import gameMgr
import dqn_agent_tetris
import collections
import matplotlib.pyplot as plt
import csv

def mean(x):
    return float(sum(x))/len(x)

showInterval = -1

if __name__=="__main__":
    gmm = gameMgr.tetris(12,6)
    epoch  = 0
    reward_history = collections.deque(maxlen=1000)
    loss_history   = collections.deque(maxlen=1000)
    n_batch = 256
    agt = dqn_agent_tetris.DQNAgent(gmm.getActionList(),"tetris", gmm.getScreenSize()[0],gmm.getScreenSize()[1])
    #agt.load("model/model.ckpt")
    fig = plt.figure(figsize=(gmm.getScreenSize()[0],gmm.getScreenSize()[1]))
    fig.canvas.set_window_title("TeTris")
    logFile = file("log","w")
    logCSV  = csv.writer(logFile)
    logCSV.writerow(["epoch","last_loss","loss_mean","last_reward","mean_reward","max_reward"])
    while True:
        epoch  += 1
        gmm.startNewEpoch()
        state_tp1, reward, _ = gmm.observe()
        terminal = False
        total_reward = 0
        while not terminal:
            state_t = state_tp1
            action = agt.select_action(state_t, epsilon=0.1)
            gmm.execute_action(action)
            state_tp1, reward, terminal = gmm.observe()
            agt.store_experience(state_t, action, reward, state_tp1, terminal)
            loss = agt.experience_replay()
            loss_history.append(loss)
            total_reward += reward
            if showInterval>0 and epoch % showInterval ==0:
                epoch = 0
                img = plt.imshow(state_tp1, interpolation="none", cmap="gray")
                fig.show()
                plt.pause(1e-10)

        reward_history.append(total_reward)
        #print "epoch=%d"%epoch,"loss=%.2e"%mean(loss_history),"reward=%2d"%reward_history[-1],"reward_avg=%.3f"%mean(reward_history),"reward_max=%d"%max(reward_history)
        print "epoch=%d"%epoch,"reward=%2d"%reward_history[-1],"reward_avg=%.3f"%mean(reward_history),"reward_max=%d"%max(reward_history)
        logCSV.writerow([epoch,reward_history[-1],mean(reward_history),max(reward_history)])
        logFile.flush()
