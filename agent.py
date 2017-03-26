import tensorflow as tf
import numpy as np
import collections,os

class agent:
    def __init__(self,actionList,inputSize,n_batch,replay_size, learning_rate, discountRate, saveFreq, saveFolder, memoryLimit):
        self.actionList = actionList
        self.n_actions = len(actionList)
        self.n_batch = n_batch
        self.inputSize = inputSize
        self.replay_size = replay_size
        self.experience  = collections.deque(maxlen=replay_size)
        self.learning_rate = learning_rate
        self.discountRate = discountRate
        self.totalCount = 0
        self.saveFreq = saveFreq
        self.saveFolder = saveFolder
        self.saveModel  = "model.ckpt"
        self.memoryLimit = memoryLimit


        self.init_model()
        return

    def leakyReLU(self,x,alpha=0.1):
        return tf.maximum(x*alpha,x)

    def init_model(self):
        # input layer (8 x 8)
        self.x = tf.placeholder(tf.float32, [None, self.inputSize])

        # flatten (64)
        x_flat = tf.reshape(self.x, [-1, self.inputSize])

        # fully connected layer (32)
        W_fc1 = tf.Variable(tf.truncated_normal([self.inputSize, self.inputSize], stddev=0.01))
        b_fc1 = tf.Variable(tf.zeros([self.inputSize]))
        h_fc1 = self.leakyReLU(tf.matmul(x_flat, W_fc1) + b_fc1)
        tf.summary.histogram("FC1_W",W_fc1)
        tf.summary.histogram("FC1_b",b_fc1)

        # fully connected layer (32)
        W_fc2 = tf.Variable(tf.truncated_normal([self.inputSize, self.inputSize], stddev=0.01))
        b_fc2 = tf.Variable(tf.zeros([self.inputSize]))
        h_fc2 = self.leakyReLU(tf.matmul(h_fc1, W_fc2) + b_fc2)
        tf.summary.histogram("FC2_W",W_fc2)
        tf.summary.histogram("FC2_b",b_fc2)

        # fully connected layer (32)
        W_fc3 = tf.Variable(tf.truncated_normal([self.inputSize, self.inputSize], stddev=0.01))
        b_fc3 = tf.Variable(tf.zeros([self.inputSize]))
        h_fc3 = self.leakyReLU(tf.matmul(h_fc2, W_fc3) + b_fc3)
        tf.summary.histogram("FC3_W",W_fc3)
        tf.summary.histogram("FC3_b",b_fc3)

        # output layer (n_actions)
        W_out = tf.Variable(tf.truncated_normal([self.inputSize, self.n_actions], stddev=0.01))
        b_out = tf.Variable(tf.zeros([self.n_actions]))
        self.y = tf.matmul(h_fc3, W_out) + b_out

        # loss function
        self.t = tf.placeholder(tf.float32, [None, self.n_actions])
        self.loss = tf.reduce_mean(tf.square(self.t - self.y))
        tf.summary.scalar("loss",self.loss)

        # train operation
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.optimizer = optimizer.minimize(self.loss)

        # saver
        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()
        if self.saveFolder: self.writer = tf.summary.FileWriter(self.saveFolder)

        # session
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=self.memoryLimit))
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())


    def selectNextAction(self, state, exploration=0.1):
        if np.random.rand() <= exploration:
            return np.random.choice(self.actionList)
        else:
            return self.actionList[np.argmax(self.sess.run(self.y, feed_dict = {self.x: [state]}))]

    def storeExperience(self, state_t, action, reward, state_tp1, terminal):
        self.experience.append((state_t, self.actionList.index(action), reward, state_tp1, terminal))
        return

    def experienceReplay(self):
        batch_x     = np.zeros([self.n_batch, self.inputSize], dtype=np.float32)
        batch_x_tp1 = np.zeros([self.n_batch, self.inputSize], dtype=np.float32)
        batch_t     = np.zeros([self.n_batch, self.n_actions], dtype=np.float32)

        batch_choice = np.random.randint(0, max(1,len(self.experience)-1) ,self.n_batch)
        for i in range(self.n_batch):
            batch_x    [i,:] = self.experience[batch_choice[i]][0] # state_t
            batch_x_tp1[i,:] = self.experience[batch_choice[i]][3] # state_tp1

        batch_t         = self.sess.run(self.y, feed_dict={self.x:batch_x})
        batch_max_state = self.sess.run(self.y, feed_dict={self.x:batch_x_tp1})

        for i in range(self.n_batch):
            state_t, actionID, reward, state_tp1, terminal = self.experience[batch_choice[i]]
            if terminal:
                batch_t[i,actionID] = reward
            else:
                batch_t[i,actionID] = reward + self.discountRate * np.max(batch_max_state[i]) # different from the original implementation


        _,loss,summary = self.sess.run([self.optimizer,self.loss,self.summary], feed_dict={self.x:batch_x,self.t:batch_t})
        if self.totalCount>0 and self.totalCount%self.saveFreq == 0:
            self.saver.save(self.sess, os.path.join(self.saveFolder,self.saveModel))
        self.totalCount += 1
        return np.mean(loss),summary

    def load(self, model_path=None):
        self.saver.restore(self.sess, model_path)
