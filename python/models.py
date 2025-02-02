import pickle
from abc import abstractmethod

import numpy as np
from gurobipy import Model, GRB


class BaseModel(object):
    """
    Base class for models, to be used as coding pattern skeleton.
    Can be used for a model on a single cluster or on multiple clusters"""

    def __init__(self):
        """Initialization of your model and its hyper-parameters"""
        pass

    @abstractmethod
    def fit(self, X, Y):
        """Fit function to find the parameters according to (X, Y) data.
        (X, Y) formatting must be so that X[i] is preferred to Y[i] for all i.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """

        print("empty fit")
        # Customize what happens in the fit function
        return

    @abstractmethod
    def predict_utility(self, X):
        """Method to call the decision function of your model

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        # Customize what happens in the predict utility function
        return

    def predict_preference(self, X, Y):
        """Method to predict which pair is preferred between X[i] and Y[i] for all i.
        Returns a preference for each cluster.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of preferences for each cluster. 1 if X is preferred to Y, 0 otherwise
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return (X_u - Y_u > 0).astype(int)

    def predict_cluster(self, X, Y):
        """Predict which cluster prefers X over Y THE MOST, meaning that if several cluster prefer X over Y, it will
        be assigned to the cluster showing the highest utility difference). The reversal is True if none of the clusters
        prefer X over Y.
        Compared to predict_preference, it indicates a cluster index.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, ) index of cluster with highest preference difference between X and Y.
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return np.argmax(X_u - Y_u, axis=1)

    def save_model(self, path):
        """Save the model in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the file in which the model will be saved
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(clf, path):
        """Load a model saved in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the path to the file to load
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


class RandomExampleModel(BaseModel):
    """Example of a model on two clusters, drawing random coefficients.
    You can use it to understand how to write your own model and the data format that we are waiting for.
    This model does not work well but you should have the same data formatting with TwoClustersMIP.
    """

    def __init__(self):
        self.seed = 444
        self.weights = self.instantiate()

    def instantiate(self):
        """No particular instantiation"""
        return []

    def fit(self, X, Y):
        """fit function, sets random weights for each cluster. Totally independant from X & Y.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        np.random.seed(self.seed)
        num_features = X.shape[1]
        weights_1 = np.random.rand(num_features) # Weights cluster 1
        weights_2 = np.random.rand(num_features) # Weights cluster 2

        weights_1 = weights_1 / np.sum(weights_1)
        weights_2 = weights_2 / np.sum(weights_2)
        self.weights = [weights_1, weights_2]
        return self

    def predict_utility(self, X):
        """Simple utility function from random weights.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        u_1 = np.dot(X, self.weights[0]) # Utility for cluster 1 = X^T.w_1
        u_2 = np.dot(X, self.weights[1]) # Utility for cluster 2 = X^T.w_2
        return np.stack([u_1, u_2], axis=1) # Stacking utilities over cluster on axis 1


class TwoClustersMIP(BaseModel):
    """Skeleton of MIP adapted to work with Gurobi."""

    def __init__(self, n_pieces, n_clusters, n_criterions, n_pairs):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature (L).
        n_clusters: int
            Number of clusters to implement in the MIP.
        n_criterions :
            Number of criterions for the utility function
        n_pairs :
            Number of samples in the data
        """
        self.seed = 123
        self.L = n_pieces
        self.K = n_clusters
        self.n = n_criterions
        self.P = n_pairs
        self.model = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables"""

        model = Model('TwoClustersMIP')
        model.setParam('Seed', self.seed)

        # Define decision variables
        self.v = model.addVars(
            [(i, l, k) for i in range(self.n) for l in range(self.L+1) for k in range(self.K)],
            vtype=GRB.CONTINUOUS, name="v", lb=0, ub=1
        )

        self.epsilon = model.addVars(
            [(j, k) for j in range(self.P) for k in range(self.K)], vtype=GRB.CONTINUOUS, name="epsilon", lb=0
        )

        # Binary assignment variables: z[j, k] = 1 if sample j is assigned to cluster k
        self.z = model.addVars(
            [(j, k) for j in range(self.P) for k in range(self.K)],
            vtype=GRB.BINARY, name="z"
        )

        return model
    
    def indicatrice_segment(self, x, xl, xl_plus_1):
        return 1 if xl <= x <= xl_plus_1 else 0


    def fit(self, X, Y):
        """Estimation of the parameters

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        print("Fitting the model")

        # Add constraints
        for j in range(self.P):
            self.model.addConstr(
                sum(self.z[j, k] for k in range(self.K)) >= 1,
                name=f"one_cluster_per_sample_{j}"
            )
        
        for k in range(self.K) :
            self.model.addConstr(
                sum(self.v[i,self.L,k] for i in range(self.n)) == 1
            )
            self.model.addConstr(
                sum(self.v[i,0,k] for i in range(self.n)) == 0
            )
            for i in range(self.n):
                for l in range(self.L):
                    self.model.addConstr(
                        self.v[i,l,k]+0.00001<=self.v[i,l+1,k]
                    )

        


        for k in range(self.K):
            for j in range(self.P) :
                self.model.addConstr(
                    sum(self.indicatrice_segment(X[j,i], l/self.L, (l+1)/self.L) * (self.v[i,l,k] + self.L*(self.v[i,l+1,k] - self.v[i,l,k]) * (X[j,i]-l/self.L)) for l in range(self.L) for i in range(self.n)) -
                    sum(self.indicatrice_segment(Y[j,i], l/self.L, (l+1)/self.L) * (self.v[i,l,k] + self.L*(self.v[i,l+1,k] - self.v[i,l,k]) * (Y[j,i]-l/self.L)) for l in range(self.L) for i in range(self.n)) +
                    self.epsilon[j,k] + 2*(1-self.z[j,k]) >= 0,
                    name = f'utility_constraint_u_{k}'
                )



        # Set the objective to minimize epsilon
        self.model.setObjective(
            sum(self.epsilon[j,k] for j in range(self.P) for k in range(self.K)), GRB.MINIMIZE
        )

        # Solve the model
        self.model.optimize()
        return

    def predict_utility(self, X):
        """
        Predict utility for each cluster based on the fitted model.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements.

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """

        
        A = np.zeros((self.P,self.K))

        for j in range(self.P) :
            for k in range(self.K) :
                A[j,k] = sum(self.indicatrice_segment(X[j,i], l/self.L, (l+1)/self.L) * (self.v[i,l,k].X + self.L*(self.v[i,l+1,k].X - self.v[i,l,k].X) * (X[j,i]-l/self.L)) for l in range(self.L) for i in range(self.n))

        return A
    
    def UTA_check(self) :
        return

class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self):
        """Initialization of the Heuristic Model.
        """
        self.seed = 123
        self.models = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables"""
        # To be completed
        return

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # To be completed
        return

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        
        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        return


