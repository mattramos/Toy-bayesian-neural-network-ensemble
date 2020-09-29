import tensorflow as tf
import numpy as np

class NN():
    def __init__(self, x_dim, y_dim, hidden_size, init_stddev_1_w, init_stddev_1_b, init_stddev_2_w, init_stddev_2_b, init_stddev_3_w, init_stddev_noise_w, learning_rate, model_bias_from_layer):
        # setting up as for a usual NN
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # set up NN
        self.inputs = tf.placeholder(tf.float32, [None, x_dim], name='inputs')
        self.modelpred = self.inputs[:, :num_models]
        self.spacetime = self.inputs[:, num_models: num_models + alpha_dim]
        self.area_weights = self.inputs[:, -1]
        self.y_target = tf.placeholder(tf.float32, [None, y_dim], name='target')
        
        self.layer_1_w = tf.layers.Dense(hidden_size, activation=tf.nn.tanh,
                                         kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_1_w),
                                         bias_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_1_b))
        self.layer_1 = self.layer_1_w.apply(self.spacetime)
        self.layer_2_w = tf.layers.Dense(num_models, activation=None,
                                         kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_2_w),
                                         bias_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_2_b))
        self.layer_2 = self.layer_2_w.apply(self.layer_1)

        self.model_coeff = tf.nn.softmax(self.layer_2)

        self.modelbias_w = tf.layers.Dense(y_dim, activation=None, use_bias=False,
                                           kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_3_w))
        if model_bias_from_layer == 1:
            self.modelbias = self.modelbias_w.apply(self.layer_1)
        elif model_bias_from_layer == 2:
            self.modelbias = self.modelbias_w.apply(self.layer_2)
            
        ### Line not right. Have added, axis=1 into the reduce sum, to make the dims work and reshaped
        # self.output now has dims which are (?, )
        self.output = tf.reduce_sum(self.model_coeff * self.modelpred, axis=1) + tf.reshape(self.modelbias, [-1])
        
        # NEW CODE NOISE
        self.noise_w = tf.layers.Dense(self.y_dim, activation=tf.nn.sigmoid, use_bias=False, #changed to sigmoid for more stable training, output always positive now
                                       kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_noise_w))
        self.noise_pred = 0.1*self.noise_w.apply(self.layer_1) #remember, sigmoid output is 0.5 centered when input is zero centered, so 0.1 makes it 0.05 centered

        # set up loss and optimiser - we'll modify this later with anchoring regularisation
        self.opt_method = tf.train.AdamOptimizer(self.learning_rate)
        
        # 22-4-20 Have reshaped noise_sq to match err_sq dims
        self.noise_sq = tf.square(self.noise_pred)[:,0] + 1e-6
        self.err_sq = tf.reshape(tf.square(self.y_target[:,0] - self.output), [-1])
        num_data_inv = tf.cast(tf.divide(1, tf.shape(self.inputs)[0]), dtype=tf.float32)

        self.mse_ = num_data_inv * tf.reduce_sum(self.err_sq) 
        self.loss_ = num_data_inv * (tf.reduce_sum(tf.divide(self.err_sq, self.noise_sq)) + tf.reduce_sum(tf.log(self.noise_sq)))
        self.optimizer = self.opt_method.minimize(self.loss_)

        return


    def get_weights(self, sess):
        '''method to return current params'''
        ops = [self.layer_1_w.kernel, self.layer_1_w.bias, self.layer_2_w.kernel, self.layer_2_w.bias, self.modelbias_w.kernel, self.noise_w.kernel]
        w1, b1, w2, b2, w3, wn = sess.run(ops)
        return w1, b1, w2, b2, w3, wn

    def anchor(self, sess, lambda_anchor):
        '''regularise around initialised parameters'''
        w1, b1, w2, b2, w3, wn = self.get_weights(sess)

        # get initial params
        self.w1_init, self.b1_init, self.w2_init, self.b2_init, self.w3_init, self.wn_init = w1, b1, w2, b2, w3, wn
        loss_anchor = lambda_anchor[0]*tf.reduce_sum(tf.square(self.w1_init - self.layer_1_w.kernel))
        loss_anchor += lambda_anchor[1]*tf.reduce_sum(tf.square(self.b1_init - self.layer_1_w.bias))
        loss_anchor += lambda_anchor[2]*tf.reduce_sum(tf.square(self.w2_init - self.layer_2_w.kernel))
        loss_anchor += lambda_anchor[3]*tf.reduce_sum(tf.square(self.b2_init - self.layer_2_w.bias))
        loss_anchor += lambda_anchor[4]*tf.reduce_sum(tf.square(self.w3_init - self.modelbias_w.kernel))
        loss_anchor += lambda_anchor[5]*tf.reduce_sum(tf.square(self.wn_init - self.noise_w.kernel)) # new param

        self.loss_anchor = tf.cast(1.0/X_train.shape[0], dtype=tf.float32) * loss_anchor
        
        # combine with original loss
        self.loss_ = self.loss_ + tf.cast(1.0/X_train.shape[0], dtype=tf.float32) * loss_anchor
        self.optimizer = self.opt_method.minimize(self.loss_)
        return


    def predict(self, x, sess):
        '''predict method'''
        feed = {self.inputs: x}
        y_pred = sess.run(self.output, feed_dict=feed)
        return y_pred
    
    def get_noise_sq(self, x, sess):
        '''get noise squared method'''
        feed = {self.inputs: x}
        noise_sq = sess.run(self.noise_sq, feed_dict=feed)
        return noise_sq

    def get_alphas(self, x, sess):
        feed = {self.inputs: x}
        alpha = sess.run(self.model_coeff, feed_dict=feed)
        return alpha
    
    def get_betas(self, x, sess):
        feed = {self.inputs: x}
        beta = sess.run(self.modelbias, feed_dict=feed)
        return beta

    def get_alpha_w(self, x, sess):
      feed = {self.inputs: x}
      alpha_w = sess.run(self.layer_2, feed_dict=feed)
      return alpha_w

    def get_w1(self, x, sess):
      feed = {self.inputs: x}
      w1 = sess.run(self.layer_1, feed_dict=feed)
      return w1

def fn_predict_ensemble(NNs,X_train, sess):
    y_pred=[]
    y_pred_noise_sq=[]
    for ens in range(len(NNs)):
        y_pred.append(NNs[ens].predict(X_train, sess))
        y_pred_noise_sq.append(NNs[ens].get_noise_sq(X_train, sess))
    y_preds_train = np.array(y_pred)
    y_preds_noisesq_train = np.array(y_pred_noise_sq)
    y_preds_mu_train = np.mean(y_preds_train,axis=0)
    y_preds_std_train_epi = np.std(y_preds_train,axis=0)
    y_preds_std_train = np.sqrt(np.mean((y_preds_noisesq_train + np.square(y_preds_train)), axis = 0) - np.square(y_preds_mu_train)) #add predicted aleatoric noise
    return y_preds_train, y_preds_mu_train, y_preds_std_train, y_preds_std_train_epi, y_preds_noisesq_train

def get_alphas(NNs, X_train, sess):
    alphas = []
    for ens in range(len(NNs)):
        alphas.append(NNs[ens].get_alphas(X_train, sess))
    return alphas


def get_betas(NNs, X_train, sess):
    betas = []
    for ens in range(len(NNs)):
        betas.append(NNs[ens].get_betas(X_train, sess))
    return betas

def get_layer2_output(NNs, X_train, sess):
    alpha_w = []
    for ens in range(len(NNs)):
        alpha_w.append(NNs[ens].get_alpha_w(X_train, sess))
    return alpha_w

def get_layer1_output(NNs, X_train, sess):
    w1 = []
    for ens in range(len(NNs)):
        w1.append(NNs[ens].get_w1(X_train, sess))
    return w1