import tensorflow as tf
import numpy as np
import math
import pandas as pd
from pylab import *
from data import *
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve
import argparse


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description='Run DAEi.')
    parser.add_argument('--path', nargs='?', default='datasets/',
                        help='Input data path.')
    parser.add_argument('--data_name', nargs='?', default='Enzyme',
                        help='Name of dataset: Enzyme, Ion Channel, GPCR, Nuclear Receptor')
    parser.add_argument('--epoches', type=int, default=300,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='Hidden size.')
    parser.add_argument('--regs', nargs='?', default='[0.000001,0.000001,0.000001]',
                        help='regularization coefficient for L2, drug-drug similarity, and target-target similarity.')
    parser.add_argument('--noise_level', type=float, default=0.00001,
                        help='Level for Gaussian noise.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--min_loss', type=float, default=0.01,
                        help='The minimum value for the loss function.')
    parser.add_argument('--cv', type=int, default=10,
                        help='K-fold Cross Validation.')
    parser.add_argument('--loss_type', nargs='?', default='square',
                        help='Type of loss function: square; cross_entropy')
    parser.add_argument('--mode', nargs='?', default='dti',
                        help='Mode for training: dti -> drug-target interaction; tdi -> target-drug interaction;')
    return parser.parse_args()


class DAEi():
    def __init__(self,
                 N=None,                  # number of drug/target
                 M=None,                  # number of target/drug
                 hidden_size=512,           # hidden layer size
                 batch_size=256,            # batch size
                 learning_rate=1e-3,         # learning rate
                 lamda_regularizer=1e-6,      # regularization coefficient for L2
                 lamda_regularizer_d=1e-6,    # regularization coefficient for drug-drug similarity
                 lamda_regularizer_t=1e-6,    # regularization coefficient for target-target similarity
                 noise_level=1e-5,         # level of Gaussian noise
                 S_d=None,               # drug-drug similarity
                 S_t=None,               # target-target similarity
                 loss_type='square',        # the type of loss function
                 mode='dti'              # model for drug-target interactions
                 ):
        self.N = N
        self.M = M
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lamda_regularizer = lamda_regularizer
        self.lamda_regularizer_d = lamda_regularizer_d
        self.lamda_regularizer_t = lamda_regularizer_t
        self.noise_level = noise_level
        self.S_d = S_d
        self.S_t = S_t
        self.loss_type = loss_type
        self.mode = mode

        self.train_loss_records = []
        self.build_graph()

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # _________ input data _________
            self.line_inputs = tf.compat.v1.placeholder(
                tf.float32, shape=[None, self.M], name='line_inputs')
            self.point_inputs = tf.compat.v1.placeholder(
                tf.int32, shape=[None, 1], name='point_inputs')
            self.S_d = tf.convert_to_tensor(self.S_d, tf.float32)
            self.S_t = tf.convert_to_tensor(self.S_t, tf.float32)

            # _________ variables _________
            self.weights = self._initialize_weights()

            # _________ train _____________
            self.y_ = self.inference(
                line_inputs=self.line_inputs, point_inputs=self.point_inputs)
            self.loss_train = self.loss_function(true_r=self.corrupted_inputs,
                                                 predicted_r=self.y_,
                                                 lamda_regularizer=self.lamda_regularizer,
                                                 loss_type=self.loss_type)
            self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                             beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss_train)

            # _________ prediction _____________
            self.predictions = self.inference(
                line_inputs=self.line_inputs, point_inputs=self.point_inputs)

            # variable init
            init = tf.compat.v1.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

    def _init_session(self):
        # adaptively growing memory
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.compat.v1.Session(config=config)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['W1'] = tf.Variable(tf.random_normal(
            [self.M, self.hidden_size], 0.0, 0.1), name='W1')
        all_weights['b1'] = tf.Variable(
            tf.zeros([self.hidden_size]), name='b1')
        all_weights['W2'] = tf.Variable(tf.random_normal(
            [self.hidden_size, self.M], 0.0, 0.1), name='W2')
        all_weights['b2'] = tf.Variable(tf.zeros([self.M]), name='b2')
        all_weights['V'] = tf.Variable(
            tf.zeros([self.N, self.hidden_size]), name='V')
        return all_weights

    def train(self, data_mat):
        instances_size = len(data_mat)
        batch_size = self.batch_size
        total_batch = math.ceil(instances_size/batch_size)
        for batch in range(total_batch):
            start = (batch*batch_size) % instances_size
            end = min(start+batch_size, instances_size)
            feed_dict = {self.point_inputs: np.reshape(data_mat[start:end, 0], (-1, 1)),
                         self.line_inputs: data_mat[start:end, 1:]}
            loss, opt = self.sess.run(
                [self.loss_train, self.train_op], feed_dict=feed_dict)
            self.train_loss_records.append(loss)

        return self.train_loss_records

    def inference(self, line_inputs, point_inputs):
        self.corrupted_inputs = line_inputs + self.noise_level * \
            tf.random_normal(tf.shape(line_inputs))
        Vu = tf.reshape(tf.nn.embedding_lookup(
            self.weights['V'], point_inputs), (-1, self.hidden_size))
        encoder = tf.nn.sigmoid(
            tf.matmul(self.corrupted_inputs, self.weights['W1']) + Vu + self.weights['b1'])
        decoder = tf.identity(
            tf.matmul(encoder, self.weights['W2']) + self.weights['b2'])
        return decoder

    def loss_function(self, true_r, predicted_r,
                      lamda_regularizer=1e-6,
                      lamda_regularizer_d=1e-6,
                      lamda_regularizer_t=1e-6,
                      loss_type='square'):
        if loss_type == 'square':
            loss = tf.compat.v1.losses.mean_squared_error(true_r, predicted_r)
        elif loss_type == 'cross_entropy':
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=true_r, logits=predicted_r)

        if lamda_regularizer > 0:
            regularizer = tf.contrib.layers.l2_regularizer(lamda_regularizer)
            regularization_1 = regularizer(self.weights['V']) + regularizer(self.weights['W1']) + regularizer(
                self.weights['W2']) + regularizer(self.weights['b1']) + regularizer(self.weights['b2'])

        if self.mode == 'dti':
            drug1 = tf.reshape(
                tf.tile(self.weights['V'], (1, self.N)), shape=[-1, self.hidden_size])
            drug2 = tf.tile(self.weights['V'], (self.N, 1))
            target1 = tf.reshape(
                tf.tile(self.weights['W1'], (1, self.M)), shape=[-1, self.hidden_size])
            target2 = tf.tile(self.weights['W1'], (self.M, 1))
        else:
            target1 = tf.reshape(
                tf.tile(self.weights['V'], (1, self.N)), shape=[-1, self.hidden_size])
            target2 = tf.tile(self.weights['V'], (self.N, 1))
            drug1 = tf.reshape(
                tf.tile(self.weights['W1'], (1, self.M)), shape=[-1, self.hidden_size])
            drug2 = tf.tile(self.weights['W1'], (self.M, 1))

        if lamda_regularizer_d > 0:
            y_score = tf.reshape(
                tf.exp(-tf.reduce_sum((drug1-drug2)*(drug1-drug2), axis=1)), shape=[-1])
            y_true = tf.reshape(self.S_d, shape=[-1])
            regularization_2 = lamda_regularizer_d * \
                tf.reduce_mean(tf.compat.v1.losses.mean_squared_error(y_true, y_score))

        if lamda_regularizer_t > 0:
            y_score = tf.reshape(
                tf.exp(-tf.reduce_sum((target1-target2)*(target1-target2), axis=1)), shape=[-1])
            y_true = tf.reshape(self.S_t, shape=[-1])
            regularization_3 = lamda_regularizer_t * \
                tf.reduce_mean(tf.compat.v1.losses.mean_squared_error(y_true, y_score))

        cost = loss + regularization_1 + regularization_2 + regularization_3
        return cost

    def predict_ratings(self, data_mat):
        pred_mat = np.zeros([self.N, self.M])

        instances_size = len(data_mat)
        batch_size = self.batch_size
        total_batch = math.ceil(instances_size/batch_size)
        for batch in range(total_batch):
            start = (batch*batch_size) % instances_size
            end = min(start+batch_size, instances_size)
            feed_dict = {self.point_inputs: np.reshape(data_mat[start:end, 0], (-1, 1)),
                         self.line_inputs: data_mat[start:end, 1:]}
            out = self.sess.run([self.predictions], feed_dict=feed_dict)
            pred_mat[start:end, :] = np.reshape(out, (-1, self.M))

        return pred_mat

    def evaluate(self, data_mat, X, labels):
        drugs_inputs = X[:, 0].astype(np.int32)
        targets_inputs = X[:, 1].astype(np.int32)
        pred_mat = self.predict_ratings(data_mat=data_mat)

        if self.mode == 'tdi':
            pred_mat = pred_mat.T
        y_pred = pred_mat[drugs_inputs, targets_inputs]

        auc_score = roc_auc_score(labels, y_pred)
        precision, recall, pr_thresholds = precision_recall_curve(
            labels, y_pred)
        aupr_score = auc(recall, precision)
        return auc_score, aupr_score


# train for model
def train(model, data_list, drugs_num, targets_num, epoches=40, cv=10, min_loss=0.01, mode='dti'):
    # k-fold cross validation
    kf = KFold(n_splits=cv, shuffle=True)
    data_mat = sequence2mat(sequence=data_list, N=drugs_num, M=targets_num)

    cv_auc_list, cv_aupr_list = [], []
    print('Train for drug-target pairs:')
    instances_list = []
    [instances_list.append([d, t, data_mat[d, t]])
     for d in range(drugs_num) for t in range(targets_num)]
    for train_ids, test_ids in kf.split(instances_list):
        train_list = np.array(instances_list)[train_ids]
        test_list = np.array(instances_list)[test_ids][:, :2]
        test_labels = np.array(instances_list)[test_ids][:, -1]
        train_mat = sequence2mat(
            sequence=train_list, N=drugs_num, M=targets_num)

        if mode == 'dti':
            point_array = np.array([d for d in range(drugs_num)])
            input_data = np.c_[point_array, train_mat]
        else:
            point_array = np.array([t for t in range(targets_num)])
            input_data = np.c_[point_array, train_mat.T]

        auc_score, aupr_score = model.evaluate(
            data_mat=input_data, X=np.array(test_list), labels=test_labels)
        print('Init: AUC = %.4f, AUPR=%.4f' % (auc_score, aupr_score))

        auc_list, aupr_list = [], []
        auc_list.append(auc_score)
        aupr_list.append(aupr_score)
        for epoch in range(epoches):
            data_mat = np.random.permutation(input_data)
            loss_records = model.train(data_mat=data_mat)
            auc_score, aupr_score = model.evaluate(
                data_mat=input_data, X=np.array(test_list), labels=test_labels)
            auc_list.append(auc_score)
            aupr_list.append(aupr_score)
            print('epoch=%d, loss=%.4f, AUC=%.4f, AUPR=%.4f' %
                  (epoch, loss_records[-1], auc_score, aupr_score))

            if (loss_records[-1] < min_loss and len(aupr_list) > 3 and
                    aupr_list[-1] <= aupr_list[-2] and aupr_list[-2] <= aupr_list[-3]):
                cv_auc = auc_list[-3]
                cv_aupr = aupr_list[-3]
                break
            cv_auc = auc_score
            cv_aupr = aupr_score
        cv_auc_list.append(cv_auc)
        cv_aupr_list.append(cv_aupr)

    print('Mean AUC=%.4f, AUPR=%.4f' %
          (np.mean(cv_auc_list), np.mean(cv_aupr_list)))


if __name__ == '__main__':
    args = parse_args()
    path = args.path
    data_name = args.data_name
    hidden_size = args.hidden_size
    lamda_regularizers = eval(args.regs)
    learning_rate = args.lr
    noise_level = args.noise_level
    batch_size = args.batch_size
    cv = args.cv
    epoches = args.epoches
    min_loss = args.min_loss
    loss_type = args.loss_type
    mode = args.mode

    # path = 'datasets'
    # data_name = 'Enzyme'       # Enzyme, Ion Channel, GPCR, Nuclear Receptor
    # hidden_size = 1024        # hidden layer size, i.e. embedding size
    # batch_size = 256         # batch size
    # learning_rate = 1e-3      # learning rate
    # lamda_regularizer = 1e-6   # regularization coefficient for L2
    # lamda_regularizer_d = 1e-6  # regularization coefficient for drug-drug similarity
    # lamda_regularizer_t = 1e-6  # regularization coefficient for target-target similarity
    # noise_level = 1e-4       # level of Gaussian noise
    # cv = 10
    # epoches = 300
    # min_loss = 0.01         # min value to stop training
    # loss_type = 'square'      # type of loss function: square; cross_entropy
    # mode = 'dti'           # dti: drug-target interactions; tdi: target-drug interactions

    data_dir = path + '/' + data_name + '.txt'
    drugs_num, targets_num, data_list, _, _ = load_data(file_dir=data_dir)
    print(data_name + ': N=%d, M=%d' % (drugs_num, targets_num))

    if data_name == 'Enzyme':
        dg_name = 'e'
        dc_name = 'e'
    elif data_name == 'Ion Channel':
        dg_name = 'ic'
        dc_name = 'ic'
    elif data_name == 'GPCR':
        dg_name = 'gpcr'
        dc_name = 'gpcr'
    elif data_name == 'Nuclear Receptor':
        dg_name = 'nr'
        dc_name = 'nr'

    dg_dir = path + '/' + dg_name + '_simmat_dg.txt'
    dc_dir = path + '/' + dc_name + '_simmat_dc.txt'
    data_dg = pd.read_table(dg_dir, sep="\t", header=0, index_col=0)
    data_dc = pd.read_table(dc_dir, sep="\t", header=0, index_col=0)
    S_d = data_dg.values
    S_t = data_dc.values

    if mode == 'dti':
        # build model
        model = DAEi(N=drugs_num,
                     M=targets_num,
                     hidden_size=hidden_size,
                     batch_size=batch_size,
                     learning_rate=learning_rate,
                     lamda_regularizer=lamda_regularizers[0],
                     lamda_regularizer_d=lamda_regularizers[1],
                     lamda_regularizer_t=lamda_regularizers[2],
                     noise_level=noise_level,
                     S_d=S_d,
                     S_t=S_t,
                     loss_type=loss_type,
                     mode=mode
                     )
    else:
        # build model
        model = DAEi(N=targets_num,
                     M=drugs_num,
                     hidden_size=hidden_size,
                     batch_size=batch_size,
                     learning_rate=learning_rate,
                     lamda_regularizer=lamda_regularizers[0],
                     lamda_regularizer_d=lamda_regularizers[1],
                     lamda_regularizer_t=lamda_regularizers[2],
                     noise_level=noise_level,
                     S_d=S_d,
                     S_t=S_t,
                     loss_type=loss_type,
                     mode=mode
                     )

    train(model=model,
          data_list=data_list,
          drugs_num=drugs_num,
          targets_num=targets_num,
          epoches=epoches,
          cv=cv,
          min_loss=min_loss,
          mode=mode)