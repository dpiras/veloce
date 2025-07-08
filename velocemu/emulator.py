import jax.numpy as np
import numpy
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
from velocemu.utils import fz_jax
# to deal with the pre-saved models 
import pkg_resources

class IntegralEmu:
    def __init__(self, dataset, prefix='dataset_pantheon_moreparams_larger_nonlin'):
        """Emulator for the velocity covariance. 
        `dataset` has to be in the form
        array([[z1, z1, 0.],
               [z2, z1, a12],
               [z2, z2, 0.],
               [z3, z1, a13],
               [z3, z2, a23],
               [z3, z3, 0.],
               [...],
               ])
        where zi corresponds to the redshuift of source i, and aij is the angle between source i and source j."""
        # extract unique rows and keep track of inverse indices
        self.unique_rows, self.inverse_indices = self._jax_unique(dataset)
        self.condition = self.unique_rows[:, 2] == 0  # check diagonal condition
        
        # constants that were used during training for pre/postprocessing
        self.C = 1e-2
        self.C_diag = 1e-5
        
        # load trained models
        model_path_offdiag = pkg_resources.resource_filename("velocemu", f"trained_models/{prefix}_offdiag.pkl")
        model_path_diag = pkg_resources.resource_filename("velocemu", f"trained_models/{prefix}_diag.pkl")
        self.cp_nn = CPJ(probe='custom', filepath=model_path_offdiag)
        self.model_parameters = ['z1', 'z2', 'alpha', 'Omat', 'H0', 'As']
        # same as before, so remove "_Z1"
        self.cp_nn_diag = CPJ(probe='custom', filepath=model_path_diag)
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