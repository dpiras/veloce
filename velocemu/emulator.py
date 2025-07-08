import jax.numpy as np
import numpy
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
from velocemu.utils import fz_jax


class IntegralEmu:
    def __init__(self, dataset, relative_path='./', prefix='velocemu/trained_models/dataset_pantheon_moreparams_larger_nonlin'):
        # extract unique rows and keep track of inverse indices
        self.unique_rows, self.inverse_indices = self._jax_unique(dataset)
        self.condition = self.unique_rows[:, 2] == 0  # check diagonal condition
        
        # constants that were used during training for pre/postprocessing
        self.C = 1e-2
        self.C_diag = 1e-5
        
        # load trained models 
        self.cp_nn = CPJ(probe='custom', filepath=f'{relative_path}/{prefix}_offdiag.pkl')
        self.model_parameters = ['z1', 'z2', 'alpha', 'Omat', 'H0', 'As']
        # same as before, so remove "_Z1"
        self.cp_nn_diag = CPJ(probe='custom', filepath=f'{relative_path}/{prefix}_diag.pkl')
        self.model_parameters_diag = ['z1', 'Omat', 'H0', 'As']
        # decided to save these as it's just easier
        np.save('./test_unique_rows', self.unique_rows)
        np.save('./test_inverse_indices', self.inverse_indices)
        np.save('./test_condition', self.condition)

    def _jax_unique(self, array):
        """JAX-compatible version of np.unique with return_inverse."""
        unique, index, inverse = np.unique(array, axis=0, return_index=True, return_inverse=True)
        return unique, inverse
        
    def predict(self, parameters, unique_rows, inverse_indices, condition, dim1, dim2):
        """The actual prediction. This uses some auxiliary files which are created when creating
        the class instance. This is such that then we can compile this as a JAX object.
        dim1 and dim2 are again some dynamic shapes to only consider the unique terms,
        in the diagonal and outside of it."""
        Omat, H0, As = parameters
        # create an empty container for the results
        integral_value = np.zeros(unique_rows.shape[0])
        
        # prepare diagonal entries
        repeats = np.tile(parameters, (dim1, 1)) # the unique diagonal terms
        unique_rows_diag = np.concatenate(
            [
                unique_rows[condition][:, :1],
                repeats,
            ],
            axis=1,
        )
        # Predict diagonal
        diag_predictions = self.cp_nn_diag.predict(unique_rows_diag) / self.C_diag
        integral_value = integral_value.at[condition].set(diag_predictions)

        # Prepare off-diagonal entries
        repeats = np.tile(parameters, (dim2, 1)) # the other, non-diagonal terms
        unique_rows_rest = np.concatenate(
            [
                unique_rows[~condition],
                repeats,
            ],
            axis=1,
        )
        # Predict rest
        rest_predictions = self.cp_nn.predict(unique_rows_rest) / self.C
        integral_value = integral_value.at[~condition].set(rest_predictions)

        # Fill in the non-unique lines
        predictions = np.take(integral_value, inverse_indices)
        
        # need to multiply by f(0)^2 due to initial choice of training models
        f0 = np.real(fz_jax(0, Omat))
        predictions *= f0**2
        return predictions