import numpy as np
import gameMgr
import agent
import collections
import matplotlib.pyplot as plt
import csv

def mean(x):
    return float(sum(x))/len(x)

def flatten(x):
    res = np.zeros(sum([r.size for r in x]),dtype=np.int32)
    pos1 = 0
    for r in x:
        pos2 = pos1 + r.size
        res[pos1:pos2] = r.flatten()[:]
        pos1 = pos2
    return res

showInterval = -1

if __name__=="__main__":
    #gmm = gameMgr.tetris(8,4)
    gmm = gameMgr.tetris(20,10)
    epoch  = 0
    write_epoch = 100
    reward_history = collections.deque(maxlen=1000)
    loss_history   = collections.deque(maxlen=1000)
    n_batch = 256
    agt = agent.agent(gmm.getActionList(),gmm.getStateSize(),n_batch,replay_size=10000)
    #agt.load("model/model.ckpt")
    fig = plt.figure(figsize=(gmm.getScreenSize()[0],gmm.getScreenSize()[1]))
    fig.canvas.set_window_title("TeTris")
    logFile = file("log","w")
    logCSV  = csv.writer(logFile)
    logCSV.writerow(["epoch","last_loss","loss_mean","last_reward","mean_reward","max_reward"])
    while True:
        epoch  += 1
        gmm.startNewEpoch()
        all_state_tp1, reward, _ = gmm.observe()
        state_tp1 = flatten(all_state_tp1)
        terminal = False
        total_reward = 0
        while not terminal:
            state_t = state_tp1
            action = agt.selectNextAction(state_t, explanation = 0.2)
            gmm.execute_action(action)
            all_state_tp1, reward, terminal = gmm.observe()
            state_tp1 = flatten(all_state_tp1)
            agt.storeExperience(state_t, action, reward, state_tp1, terminal)
            loss,summary = agt.experienceReplay()
            loss_history.append(loss)
            total_reward += reward
            if epoch%write_epoch==0:
                agt.writer.add_summary(summary,epoch)
            if showInterval>0 and epoch % showInterval ==0:
                epoch = 0
                img = plt.imshow(all_state_tp1[0]+all_state_tp1[1], interpolation="none", cmap="gray")
                fig.show()
                plt.pause(1e-10)

        reward_history.append(total_reward)
        print "epoch=%d"%epoch,"loss=%.2e"%mean(loss_history),"reward=%2d"%reward_history[-1],"reward_avg=%.3f"%mean(reward_history),"reward_max=%d"%max(reward_history)
        logCSV.writerow([epoch,loss_history[-1],mean(loss_history),reward_history[-1],mean(reward_history),max(reward_history)])
        logFile.flush()
