import numpy as np

class BinaryRBM:
    def __init__(self, hid_units, learning_rate = 0.1, epochs = 20, batch_size = 100, cdk = 1, persistent = False,
                 measure = 'reconstruction_error'):
        self.num_hid = hid_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.cdk = cdk
        self.persistent = persistent
        if measure == 'reconstruction_error':
            self.measure_progress = self._get_recon_error
        elif measure == 'pseudo_likelihood':
            self.measure_progress = self._get_pseudo_likelihood
        else:
            raise ValueError
        self.hid_prob = None
        
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
                score += self.measure_progress(vis_state)
            self._log(score)
            
    def _run_gibbs(self, k):
        for i in range(k):
            hid_state = self.hid_prob > np.random.uniform(size = self.hid_prob.shape) 
            self.vis_prob = self.vis_from_hid(hid_state)
            vis_state = self.vis_prob > np.random.uniform(size = self.vis_prob.shape)
            self.hid_prob = self.hid_from_vis(vis_state)
        
    def classify(self, data):
        vis_bin = np.concatenate(data, np.zeros((data.shape[0], self.label_count)), axis = 1)
        hid_prob = self.hid_from_vis(vis_bin)
        hid_bin = hid_prob > np.random.uniform(size = hid_prob.shape)
        vis_prob = self.vis_from_hid(hid_bin)
        vis_bin = vis_prob > np.random.uniform(size = vis_prob.shape)
        return np.argmax(vis_bin[:, self.data.shape[1]:])
     
    def _generate_minibatch(self):
        num_train = self.data.shape[0]
        num_batches = num_train // self.batch_size
        while True:
            shuffled_ind = np.random.permutation(num_train)
            for i in range(0, num_batches, self.batch_size):
                ind = shuffled_ind[i:i+batch_size]
                yield self.data[ind, :], None if self.labels is None else self.labels[ind]

    def _concat_units(self, data, labels):
        total_labels = labels.shape[0]
        label_units = np.zeros((total_labels, self.label_count))
        label_units[np.arange(total_labels), labels] = 1
        return np.concatenate((data, label_units), axis = 1)
    
    def _get_free_energy(self, X):
        return -np.dot(X, self.vis_bias.T) - np.logsumexp(0, np.dot(X, self.weights) + self.hid_bias)
    
    def _get_recon_error(self, vis_state):
        return np.sum((vis_state - self.vis_prob) ** 2) / self.batch_size
        
    def _get_pseudo_likelihood(self, vis_state):
        pass
        
    def _log(self, score):
        pass

    def hid_from_vis(self, vis_state):
        return self.sigmoid(np.dot(vis_state, self.weights) + self.hid_bias)
        
    def vis_from_hid(self, hid_state):
        return self.sigmoid(np.dot(hid_state, self.weights.T) + self.vis_bias)
        
    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))
