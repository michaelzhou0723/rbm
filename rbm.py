import numpy as np
import pickle
from utils import normalise_data

class BinaryRBM:
    def __init__(self, hid_units, learning_rate = 0.1, epochs = 5, batch_size = 100, weight_decay = 0.0001, momentum = 0.5, 
                 cdk = 1, persistent = True, measure = 're', save = True):
        """
        A Generative Restricted Boltzmann Machine with binary visible and hidden units.
        
        Training can be done by Contrastive Divergence or Persistent Contrastive Divergence. Additional 
        hyperparameters such as weight decay constant and the momentum can be fine-tuned for best result.
        
        After training, this model can classify unlabelled data and reconstruct unseen data with the learned
        distribution.
        
        
        Parameters
        ----------
        hid_units: Number of hidden layer units.
        
        learning_rate: The learning rate for gradient asecnt.
        
        epochs: Number of iterations the entire training data is used.
        
        batch_size: Number of training samples in a minibatch.
        
        weight_decay: Weight decay constant for gradient ascent. Adding a 
                      weight decay term helps improve the mixing rate of 
                      the underlying Markov chain.
        
        momentum: Momentum constant for gradient ascent. Typical values
                  for RBM lie in [0.5, 0.9].
                  
        cdk: Number of steps for Gibbs block sampling. Hinton has proved
             that even CD-1 is a good approximation to the model distriution.
             
        persistent: Use Persistent Contrastive Divergence or not. Training by
                    PCD typically results in better generative models, but can
                    suffer from divergence.
                    
        measure: Measure of progress. Choose 're' for reconstruction error and
                 'pl' for pseudo-likelihood.
                 
        save: Whether save the model after training. The default path is "./trained_rbm".
        
        """
        self.num_hid = hid_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.cdk = cdk
        self.persistent = persistent
        self.weight_decay = weight_decay
        self.momentum = momentum
        if measure not in ('re', 'pl'):
            raise ValueError
        self.measure = measure
        self.save = save
        
    def _init_params(self, data, labels):
        self.data = normalise_data(data)
        self.labels = labels
        self.label_count = np.unique(labels).size
        self.num_fea = data.shape[1]
        self.num_vis = self.num_fea + self.label_count
        self.weights = np.random.normal(0, 0.01, (self.num_vis, self.num_hid))
        self.vis_bias = np.zeros((1, self.num_vis))
        self.hid_bias = np.zeros((1, self.num_hid))       
        
    def train(self, data, labels):
        """
        Parameters
        ----------
        data: A 2d numpy array with training samples in rows and features in columns.
        
        labels: A 1d numpy array containing labels corresponding to the training data.
        
        """
        self._init_params(data, labels)
        self.hid_prob = None
        
        batch_gen = self._generate_minibatch()
        num_iter = data.shape[0] // self.batch_size
        
        w_update_prev = vb_update_prev = hb_update_prev = 0
        for i in range(self.epochs):
            score = 0
            for j in range(num_iter):
                bdata, blabels = next(batch_gen)
                
                # positive phase
                vis_state = self._concat_units(bdata, blabels)
                hid_prob = self.hid_from_vis(vis_state)
                if not self.persistent or self.hid_prob is None:
                    self.hid_prob = hid_prob
                    
                # negative phase
                self.vis_prob, self.hid_prob = self.run_gibbs(self.hid_prob, self.cdk)
                
                dw = ((np.dot(vis_state.T, hid_prob) - np.dot(self.vis_prob.T, self.hid_prob)) / self.batch_size
                      - self.weight_decay * self.weights)
                dvb = np.mean(vis_state - self.vis_prob, axis = 0)
                dhb = np.mean(hid_prob - self.hid_prob, axis = 0)
                
                w_update = self.learning_rate * dw + self.momentum * w_update_prev
                vb_update = self.learning_rate * dvb + self.momentum * vb_update_prev
                hb_update = self.learning_rate * dhb + self.momentum * hb_update_prev
                
                self.weights += w_update
                self.vis_bias += vb_update
                self.hid_bias += hb_update
                
                if self.measure == 're':
                    score += self._get_recon_error(vis_state)
                elif self.measure == 'pl':
                    score += self._get_pseudo_likelihood()
                    
                w_update_prev = w_update
                vb_update_prev = vb_update
                hb_update_prev = hb_update
            self._log_progress(i, score)
            
        if self.save:
            with open(r'trained_rbm', 'wb') as f:
                pickle.dump(self, f)
            
    def run_gibbs(self, hid_prob, transitions):
        for i in range(transitions):
            hid_state = hid_prob > np.random.uniform(size = hid_prob.shape) 
            vis_prob = self.vis_from_hid(hid_state)
            vis_state = vis_prob > np.random.uniform(size = vis_prob.shape)
            hid_prob = self.hid_from_vis(vis_state)
        return vis_prob, hid_prob
        
    def classify(self, data):
        """
        Parameters
        ----------
        data: A 2d numpy array with training samples in rows and features in columns.
        
   
        Return Values:
        ----------
        A 1d numpy array containing labels corresponding to the training data.
        
        """
        vis_state = normalise_data(data)
        hid_prob = self.sigmoid(np.dot(vis_state, self.weights[np.arange(self.num_fea), :]) + self.hid_bias)
        hid_state = hid_prob > np.random.uniform(size = hid_prob.shape)
        lbl_prob = self.sigmoid(np.dot(hid_state, self.weights.T[:, self.num_fea:]) + self.vis_bias[:, self.num_fea:])
        return np.argmax(lbl_prob, axis = 1)
        
    def reconstruct(self, data, labels, num_iter = 100):
        """
        Parameters
        ----------
        data: A 2d numpy array with training samples in rows and features in columns.
        
        labels: A 1d numpy array containing labels corresponding to the training data.
        
        
        Return Values:
        ----------
        A 2d numpy array with reconstructed samples in rows and features in columns.
        
        """
        vis_state = self._concat_units(normalise_data(data), labels)
        hid_prob = self.hid_from_vis(vis_state)
        _, hid_prob = self.run_gibbs(hid_prob, num_iter - 1)
        hid_state = hid_prob > np.random.uniform(size = hid_prob.shape)
        vis_prob = self.vis_from_hid(hid_state)
        return vis_prob[:, np.arange(self.num_fea)]
        
    def _generate_minibatch(self):
        num_train = self.data.shape[0]
        num_batches = num_train // self.batch_size
        while True:
            shuffled_ind = np.random.permutation(num_train)
            for i in range(0, num_batches, self.batch_size):
                ind = shuffled_ind[i:i+self.batch_size]
                yield self.data[ind, :], self.labels[ind]

    def _concat_units(self, data, labels):
        total_labels = labels.shape[0]
        label_units = np.zeros((total_labels, self.label_count))
        label_units[np.arange(total_labels), labels] = 1
        return np.concatenate((data, label_units), axis = 1)
    
    def _get_free_energy(self, v):
        return -(np.dot(v, self.vis_bias.T).reshape((v.shape[0],)) 
                 + np.logaddexp(0, np.dot(v, self.weights) + self.hid_bias).sum(axis = 1))
    
    def _get_recon_error(self, v):
        return np.sum((v - self.vis_prob) ** 2)
        
    def _get_pseudo_likelihood(self):
        v = self.vis_prob > np.random.uniform(size = self.vis_prob.shape)
        
        # randomly flip one unit in each sample
        rand_ind = np.random.randint(0, self.num_vis, (v.shape[0],))
        u = v.copy()
        u[np.arange(u.shape[0]), rand_ind] = 1 - u[np.arange(u.shape[0]), rand_ind]
        return self.num_vis * np.log(self.sigmoid(self._get_free_energy(u) - self._get_free_energy(v))).sum()
        
    def _log_progress(self, epoch, score):
        width = len(str(self.epochs))
        print('Epoch {}:  '.format(epoch + 1), end = '')
        if self.measure == 're':
            print('Reconstruction Error {:0{w}}'.format(score, w = width))
        else:
            print('Pseudo Likelihood {:0{w}}'.format(score, w = width))
            
    def hid_from_vis(self, v):
        return self.sigmoid(np.dot(v, self.weights) + self.hid_bias)
        
    def vis_from_hid(self, h):
        return self.sigmoid(np.dot(h, self.weights.T) + self.vis_bias)
        
    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))
        
    def __getstate__(self):
        state = self.__dict__.copy()
        
        # delete obsolete numpy arrays when pickling
        for attr in ('data', 'labels', 'vis_prob', 'hid_prob'):
            if attr in state:
                del state[attr]
        return state
        
    
