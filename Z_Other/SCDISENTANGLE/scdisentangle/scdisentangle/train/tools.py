from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import torch
import numpy as np
import math
import random
from icecream import ic

def set_seed(seed):
    ic('Setting seed to', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_stacked_inputs(inputs_list):
        if len(inputs_list) == 0:
            return torch.tensor(inputs_list)
        else:
            return torch.cat(inputs_list, dim=1)

def get_summed_inputs(inputs_list):

    if len(inputs_list) == 0:

        tensor = torch.tensor(inputs_list)
    else:
        tensor = torch.sum(torch.stack(inputs_list, dim=0), dim=0)

    return tensor

def train_sklearn(X, y, model='mlp', n_neurons=100, X_ood=None, y_ood=None):
    """
    Train MLP classifier and return accuracy on test_set
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
        )

    if model == 'mlp':
        predictor = MLPClassifier
    elif model == 'logreg':
        predictor = LogisticRegression 

    if model == 'logreg':
        clf = predictor(
        random_state=1, 
        max_iter=200, 
        verbose = False
        ).fit(X_train, y_train)
    else:
        clf = predictor(
            hidden_layer_sizes=(n_neurons,), 
            random_state=1, 
            max_iter=200, 
            verbose = False
            ).fit(X_train, y_train)

    cls_acc = clf.score(X_test, y_test)

    if X_ood is not None:
        cls_ood = clf.score(X_ood, y_ood)

        return cls_acc, cls_ood

    return cls_acc

def normalize_log(data, target_sum=None, log=True, axis=1, inplace=False):

    if not inplace:
        data = data.copy()
    
    sums = data.sum(axis=axis, keepdims=True)
    
    if target_sum is None:
        target_sum = np.median(sums)
    
    sums = np.maximum(sums, 1e-12)
    data *= target_sum / sums
    
    if log:
        data = np.log1p(data)
    return data

def mutual_information_discrete(z, c, n_bins):
    """Estimate mutual information using histogram binning."""
    # Digitize continuous data into bins
    
    z_binned = np.digitize(z, bins=np.linspace(np.min(z), np.max(z), n_bins))
    return mutual_info_score(z_binned, c)

def empirical_entropy(labels):
    """Calculate the empirical entropy of labels."""
    values, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

def compute_mig(latent_space, labels):
    """Compute the Mutual Information Gap (MIG) for latent variables with respect to a label."""
    n_bins = int(math.sqrt(latent_space.shape[0]))
    #print('Number of bins', n_bins)
    n_features = latent_space.shape[1]
    mi_scores = np.array([mutual_information_discrete(latent_space[:, i], labels, n_bins=n_bins) for i in range(n_features)])
    
    # Sort the mutual information scores in descending order
    sorted_mi_scores = np.sort(mi_scores)[::-1]
    
    # Calculate MIG as the difference between the highest MI and the second highest MI
    if len(sorted_mi_scores) > 1:
        mig = (sorted_mi_scores[0] - sorted_mi_scores[1]) / empirical_entropy(labels)
    else:
        mig = 0  # If there's only one feature, MIG does not apply

    return mig

def compute_mutual_information(tensor, estimation='continuous', n_bins=None):
    """
    Computes the mutual information between each component of a tensor with continuous variables.
    
    Parameters:
    - tensor: A numpy array of shape (n_samples, d) representing the input data.
    
    Returns:
    - A numpy array of shape (d, d) containing the mutual information values between each pair of components.
    """
    n_samples, d = tensor.shape
    mi_matrix = np.zeros((d, d))
    
    for i in range(d):
        for j in range(d):
            if i == j:
                # For mutual information of a variable with itself, we can set this as NaN or some placeholder,
                # since it's not meaningful to compute it in this context.
                mi_matrix[i, j] = np.nan
            else:
                # Reshape the data for mutual_info_regression, which expects a 2D array for X
                X = tensor[:, i].reshape(-1, 1)
                Y = tensor[:, j]
                if estimation == 'continuous':
                    mi = mutual_info_regression(X, Y)
                elif estimation == 'discrete':
                    if n_bins == None:
                        n_bins = int(math.sqrt(tensor.shape[0]))
                    mi = mutual_information_discrete(X, Y.squeeze(), n_bins=n_bins)
                else:
                    raise ValueError('estimation in compute_mutual_information either continuous or discrete')
                mi_matrix[i, j] = mi[0]  # mutual_info_regression returns an array of MI values for each feature in X
                
    return mi_matrix
    
def get_trainer(hparams, wandb_log=False, seed_nb=42):
    from hygeia.data.load_rnaseq import DatasetLoader
    from scdisentangle.train.trainer import Trainer
    
    set_seed(seed_nb)
      
    if isinstance(hparams, str):
        import yaml
        with open(hparams, 'r') as stream:
            hparams = yaml.safe_load(stream)
    
    if wandb_log:
        hparams['wandb']['wandb_log'] = True
    else:
        hparams['wandb']['wandb_log'] = False
        
    dataset = DatasetLoader(
        path=hparams['data']['file_path'],
        label_keys=hparams['data']['label_keys'],
        default_normalization=hparams['data']['default_normalization'],
        min_gene_counts=hparams['data']['min_gene_counts'],
        min_cell_counts=hparams['data']['min_cell_counts'],
        n_highly_variable_genes=hparams['data']['highly_variable'],
        use_counts = hparams['data']['use_counts'],
        subset=hparams['data']['SUBSET'],
    )

    # Dataloader contu containing all data samples
    dataloader = dataset.get_dataloader(batch_size=2000)

    # Train test val split
    train_dataloader, val_dataloader, test_dataloader = dataset.train_val_test(
        test_size=hparams['data']['test_size'],
        val_size=hparams['data']['val_size'],
        batch_size=hparams['data']['batch_size'],
    )
    
    trainer = Trainer(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        dataloader=dataloader,
        dataset=dataset,
        device=hparams['hardware']['device'],
        hparams=hparams,
    )
    
    return trainer
