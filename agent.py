import tensorflow as tf
import numpy as np
import collections

class agent:
    def __init__(self,actionList,screen_n_rows,screen_n_cols,n_batch,replay_size):
        self.actionList = actionList
        self.n_actions = len(actionList)
        self.n_batch = n_batch
        self.screen_n_rows = screen_n_rows
        self.screen_n_cols = screen_n_cols
        self.replay_size = replay_size
        self.experience  = collections.deque(maxlen=replay_size)
        self.learning_rate = 1e-4
        self.discount = 0.9
        self.totalCount = 0
        self.saveFreq = 1000
        self.safeName = "model/model.ckpt"

        self.init_model()
        return

    def init_model(self):
        # input layer (8 x 8)
        self.x = tf.placeholder(tf.float32, [None, self.screen_n_rows, self.screen_n_cols])

        # flatten (64)
        x_flat = tf.reshape(self.x, [-1, self.screen_n_rows*self.screen_n_cols])

        # fully connected layer (32)
        W_fc1 = tf.Variable(tf.truncated_normal([self.screen_n_rows*self.screen_n_cols, self.screen_n_rows*self.screen_n_cols], stddev=0.01))
        b_fc1 = tf.Variable(tf.zeros([self.screen_n_rows*self.screen_n_cols]))
        h_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)

        # fully connected layer (32)
        W_fc2 = tf.Variable(tf.truncated_normal([self.screen_n_rows*self.screen_n_cols, self.screen_n_rows*self.screen_n_cols], stddev=0.01))
        b_fc2 = tf.Variable(tf.zeros([self.screen_n_rows*self.screen_n_cols]))
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

        # fully connected layer (32)
        W_fc3 = tf.Variable(tf.truncated_normal([self.screen_n_rows*self.screen_n_cols, self.screen_n_rows*self.screen_n_cols], stddev=0.01))
        b_fc3 = tf.Variable(tf.zeros([self.screen_n_rows*self.screen_n_cols]))
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

        # output layer (n_actions)
        W_out = tf.Variable(tf.truncated_normal([self.screen_n_rows*self.screen_n_cols, self.n_actions], stddev=0.01))
        b_out = tf.Variable(tf.zeros([self.n_actions]))
        self.y = tf.matmul(h_fc3, W_out) + b_out

        # loss function
        self.t = tf.placeholder(tf.float32, [None, self.n_actions])
        self.loss = tf.reduce_mean(tf.square(self.t - self.y))

        # train operation
        #optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.optimizer = optimizer.minimize(self.loss)

        # saver
        self.saver = tf.train.Saver()

        # session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def selectNextAction(self, state, explanation=0.1):
        if np.random.rand() <= explanation:
            return np.random.choice(self.actionList)
        else:
            return self.actionList[np.argmax(self.sess.run(self.y, feed_dict = {self.x: [state]}))]

    def storeExperience(self, state_t, action, reward, state_tp1, terminal):
        self.experience.append((state_t, self.actionList.index(action), reward, state_tp1, terminal))
        return

    def experienceReplay(self):
        batch_x     = np.zeros([self.n_batch, self.screen_n_rows, self.screen_n_cols], dtype=np.float32)
        batch_x_tp1 = np.zeros([self.n_batch, self.screen_n_rows, self.screen_n_cols], dtype=np.float32)
        batch_t     = np.zeros([self.n_batch, self.n_actions], dtype=np.float32)

        batch_choice = np.random.randint(0, max(1,len(self.experience)-1) ,self.n_batch)
        for i in range(self.n_batch):
            batch_x    [i,:] = self.experience[batch_choice[i]][0] # state_t
            batch_x_tp1[i,:] = self.experience[batch_choice[i]][3] # state_tp1

        batch_max_state = self.sess.run(self.y, feed_dict={self.x:batch_x_tp1})

        for i in range(self.n_batch):
            state_t, actionID, reward, state_tp1, terminal = self.experience[batch_choice[i]]
            #batch_x[i,:] = state_t
            if terminal:
                batch_t[i,actionID] = reward
            else:
                #batch_t[i,np.argmax(batch_max_state[i])] = reward + self.discount * np.max(batch_max_state[i]) # different from the original implementation
                batch_t[i,actionID] = reward + self.discount * np.max(batch_max_state[i]) # different from the original implementation

        _,loss = self.sess.run([self.optimizer,self.loss], feed_dict={self.x:batch_x,self.t:batch_t})
        #print(loss)
        if self.totalCount>0 and self.totalCount%self.saveFreq == 0:
            self.saver.save(self.sess, "model.ckpt")
        self.totalCount += 1
        return np.mean(loss)

    def load(self, model_path=None):
        self.saver.restore(self.sess, model_path)
