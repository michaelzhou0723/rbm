import numpy as np

class BinaryRBM:
    def __init__(self, hid_units, learning_rate = 0.1, epochs = 20, batch_size = 100, cdk = 1, persistent = False,
                 measure = 're'):
        self.num_hid = hid_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.cdk = cdk
        self.persistent = persistent
        if measure not in ('re', 'pl'):
            raise ValueError
        self.measure = measure
        
    def _feed(self, data, labels = None):
        self.data = data
        self.labels = labels
        if labels is None:
            self.label_count = 0
        else:
            self.label_count = np.unique(labels).size 
        self.num_vis = data.shape[1] + self.label_count
        self.weights = np.random.normal(0, 0.01, (self.num_vis, self.num_hid))
        self.vis_bias = np.zeros((1, self.num_vis))
        self.hid_bias = np.zeros((1, self.num_hid))        
        
    def train(self, data, labels = None):
        self._feed(data, labels)
        self.hid_prob = None
        batch_gen = self._generate_minibatch()
        num_iter = data.shape[0] // self.batch_size
        for i in range(self.epochs):
            score = 0
            for j in range(num_iter):
                bdata, blabels = next(batch_gen)
                if self.labels is None:
                    vis_state = bdata
                else:
                    vis_state = self._concat_units(bdata, blabels)
                hid_prob = self.hid_from_vis(vis_state)
                hid_state = hid_prob > np.random.uniform(size = hid_prob.shape)
                if not self.persistent or self.hid_prob is None:
                    self.hid_prob = hid_prob
                self._run_gibbs(self.cdk)
                dw = (np.dot(vis_state.T, hid_prob) - np.dot(self.vis_prob.T, self.hid_prob)) / self.batch_size
                dvb = np.mean(vis_state - self.vis_prob, axis = 0)
                dhb = np.mean(hid_prob - self.hid_prob, axis = 0)
                self.weights += self.learning_rate * dw
                self.vis_bias += self.learning_rate * dvb
                self.hid_bias += self.learning_rate * dhb
                if self.measure == 're':
                    score += self._get_recon_error(vis_state)
                elif self.measure == 'pl':
                    score += self._get_pseudo_likelihood()
            self._log_progress(i, score)
            
    def _run_gibbs(self, k):
        for i in range(k):
            hid_state = self.hid_prob > np.random.uniform(size = self.hid_prob.shape) 
            self.vis_prob = self.vis_from_hid(hid_state)
            vis_state = self.vis_prob > np.random.uniform(size = self.vis_prob.shape)
            self.hid_prob = self.hid_from_vis(vis_state)
        
    def classify(self, data):
        if self.labels is None:
            raise ValueError
        vis_bin = np.concatenate((data, np.zeros((data.shape[0], self.label_count))), axis = 1)
        hid_prob = self.hid_from_vis(vis_bin)
        hid_bin = hid_prob > np.random.uniform(size = hid_prob.shape)
        vis_prob = self.vis_from_hid(hid_bin)
        vis_bin = vis_prob > np.random.uniform(size = vis_prob.shape)
        return np.argmax(vis_bin[:, self.data.shape[1]:], axis = 1)
     
    def _generate_minibatch(self):
        num_train = self.data.shape[0]
        num_batches = num_train // self.batch_size
        while True:
            shuffled_ind = np.random.permutation(num_train)
            for i in range(0, num_batches, self.batch_size):
                ind = shuffled_ind[i:i+self.batch_size]
                yield self.data[ind, :], None if self.labels is None else self.labels[ind]

    def _concat_units(self, data, labels):
        total_labels = labels.shape[0]
        label_units = np.zeros((total_labels, self.label_count))
        label_units[np.arange(total_labels), labels] = 1
        return np.concatenate((data, label_units), axis = 1)
    
    def _get_free_energy(self, v):
        return -np.dot(v, self.vis_bias.T) - np.logaddexp(0, np.dot(v, self.weights) + self.hid_bias).sum(axis = 1)
    
    def _get_recon_error(self, v):
        return np.sum((v - self.vis_prob) ** 2) / self.batch_size
        
    def _get_pseudo_likelihood(self):
        v = self.vis_prob > np.random.uniform(size = self.vis_prob.shape)
        rand_ind = np.random.randint(0, self.num_vis, (v.shape[0],))
        u = v.copy()
        u[np.arange(u.shape[0]), rand_ind] = 1 - u[np.arange(u.shape[0]), rand_ind]
        return self.num_vis * np.log(self.sigmoid(self._get_free_energy(u) - self._get_free_energy(v))).sum()
        
    def _log_progress(self, epoch, score):
        print('Epoch {}:  '.format(epoch + 1), end = '')
        if self.measure == 're':
            print('Reconstruction Error {}'.format(score))
        else:
            print('Pseudo Likelihood {}'.format(score))
            
    def hid_from_vis(self, v):
        return self.sigmoid(np.dot(v, self.weights) + self.hid_bias)
        
    def vis_from_hid(self, h):
        return self.sigmoid(np.dot(h, self.weights.T) + self.vis_bias)
        
    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))
        
    
