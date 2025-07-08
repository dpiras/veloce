# some utilities, some of them taken from scoutpip:
# https://github.com/fsorrenti/scoutpip/tree/main
# Hopefully, the meaning of all terms is self-explanatory. 

import numpy as np
from scipy import special, integrate
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln


def amplitude_W(r1, r2, alpha):
    """Amplitude term going into the window function definition."""
    return np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(alpha))
    
def window_function_w(k, r1, r2, alpha):
    """Implementation of Eq. A11 of 1010.4276."""
    if r1 == r2 and alpha == 0:
        return 1/3 # note this is 1/3, as per the theory

    A = amplitude_W(r1, r2, alpha)
    j0 = special.spherical_jn(0, k * A)
    j2 = special.spherical_jn(2, k * A)
    return (
        np.cos(alpha) * (j0 - 2 * j2) / 3 + j2 * r1 * r2 * (np.sin(alpha))**2 / A**2
    )

def omm(z, omegam0):
    """Matter density as a function of z and \Omega_m(z=0)."""
    return (omegam0 * (1 + z) ** 3) / (1 + omegam0 * z * (3 + z * (3 + z)))
    
def fz(z, omegam0):
    """Growth rate as a function of z and \Omega_m(z=0)."""
    return (
        omm(z, omegam0)
        * (5 / special.hyp2f1(1 / 3, 1, 11 / 6, (1 - 1 / omm(z, omegam0))) - 3)
        / 2
    )

def Hz(z, H0, omegam0):
    """Hubble parameter as a function of z, H_0 and \Omega_m(z=0)."""
    return H0 * ((1 - omegam0) + omegam0 * (1 + z) ** 3) ** (1 / 2)

def D1z(z, omegam0):
    """Growth function as a function of z and \Omega_m(z=0)."""
    return (
        6
        * special.hyp2f1(1 / 3, 1, 11 / 6, (1 - 1 / omm(z, omegam0)))
        / (5 * omegam0 * (1 + z))
    )

def B_factor(z, H0, Omat):   
    """Prefactor product."""
    return (fz(z, Omat) * Hz(z, H0, Omat)) / ((1 + z))

def prefactor_divergence(z1, z2, H0, Omat):
    """Prefactor product."""
    B1 = B_factor(z1, H0, Omat)
    B2 = B_factor(z2, H0, Omat)
    return B1 * B2

def integrand_ztor(z, H0, Omat):
    """Integrand of the luminosity_distance function below."""
    return 1 / (H0 * np.sqrt(Omat * (1 + z) ** 3 + (1 - Omat)))
    
def luminosity_distance(z, H0, Omat):
    """Basic luminosity distance function."""
    c = 299792.458
    return (
        c * (1 + z) * integrate.quad(integrand_ztor, 0, z, args=(H0, Omat), limit=1000000, epsabs=1.49e-12, epsrel=1.49e-12)[0]
    )  

def B_nonlin_prefac(k, sigma8):
    """Non-linear damping term, from https://arxiv.org/pdf/1809.09338."""
    a1 = -0.817 + 3.198*sigma8
    a2 = 0.877 - 4.191*sigma8
    a3 = -1.199 + 4.629*sigma8
    exponential = np.maximum(a1 + a2*k + a3*k**2, 0) # to avoid positive values which are not being damped
    return np.exp(-k * exponential)
    
def D_nonlin_prefac(k, sigmanu=13):
    """Non-linear term in sinc(x) form, from https://arxiv.org/pdf/1312.1022."""
    return np.sin(k*sigmanu) / (k*sigmanu)

def hyp2f1_continuation(a,b,c,z):
    """JAX version of the hypergeometrical(2,1) function.
    This was not available directly in JAX when preparing this paper.
    It might become available in a better form in a future version."""
    # d0 = 1 and d_{-1} = 0
    prev_da = 1.
    prev_db = 1.
    prev_prev_da = 0.
    prev_prev_db = 0.

    # partial_sum_1 corresponds to the summation on the top line of equation 4.21
    # partial_sum_2 corresponds to the summation on the bottom line of equation 4.21
    partial_sum_1 = 1.
    partial_sum_2 = 1.

    # If z is on the branch cut, take the value above the branch cut
    z = jnp.where(jnp.imag(z) == 0., jnp.where(jnp.real(z)>=1., z + 0.0000001j, z), z)

    def body_fun(j, val):
        a_, b_, c_, z_, prev_prev_da, prev_prev_db, prev_da, prev_db, partial_sum_1, partial_sum_2 = val

        #------------------------------------------------------------------------------------------------------
        # This section of the function handles the summation on the first line of equation 4.21
        # calculates d_j and the corresponding term in the sum
        d_ja = (j+a_-1.)/(j*(j+a_-b_)) * (((a_+b_+1.)*0.5-c_)*prev_da + 0.25*(j+a_-2.)*prev_prev_da)
        partial_sum_1 += d_ja * (z - 0.5)**(-j)

        # updates d_{j-2} and d_{j-1}
        prev_prev_da = prev_da
        prev_da = d_ja
        #------------------------------------------------------------------------------------------------------
        # This section of the function handles the summation on the second line of equation 4.21
        # calculates d_j and the corresponding term in the sum
        d_jb = (j+b_-1.)/(j*(j-a_+b_)) * (((a_+b_+1)*0.5-c_)*prev_db + 0.25*(j+b_-2.)*prev_prev_db)
        partial_sum_2 += d_jb * (z - 0.5)**(-j)

        # updates d_{j-2} and d_{j-1}
        prev_prev_db = prev_db
        prev_db = d_jb

        return [a, b, c, z, prev_prev_da, prev_prev_db, prev_da, prev_db, partial_sum_1, partial_sum_2]

    result = jax.lax.fori_loop(1, 30, body_fun, [a, b, c, z, prev_prev_da, prev_prev_db, prev_da, prev_db, partial_sum_1, partial_sum_2])

    # includes the gamma function prefactors in equation 4.21 to compute the final result of 2F1
    final_result = gamma(c) * (result[8] * gamma(b-a)/gamma(b)/gamma(c-a)*(0.5-z)**(-a) + \
                               result[9] * gamma(a-b)/gamma(a)/gamma(c-b)*(0.5-z)**(-b)
                              )
    return final_result

def gamma(val):
    """Utility function for the JAX version of hyp21 function."""
    return jnp.where(val >= 0, jnp.exp(gammaln(val)), -jnp.exp(gammaln(val)))


def fz_jax(z, omegam0):
    """Growth rate as a function of z and \Omega_m(z=0), as a JAX version.
    This is needed since technically the trained models have this factor built-in
    due to an initial choice of the training function, which was different with
    respect to the final one."""    
    return (
        omm(z, omegam0)
        * (5 / hyp2f1_continuation(1 / 3, 1, 11 / 6, (1 - 1 / omm(z, omegam0))) - 3)
        / 2
    )

def reconstruct_symmetric_matrix_from_lower_diagonal(array, len_matrix):
    """Reconstruct a symmetrix matrix from its lower diagonal."""
    
    # Step 1: Initialize a new matrix of the same size with zeros
    reconstructed_matrix = jnp.zeros((len_matrix, len_matrix))

    # Step 2: Get the indices for the lower triangular matrix including the diagonal
    indices = jnp.tril_indices(len_matrix)

    # Step 3: Fill the lower part of the matrix from the array
    reconstructed_matrix = reconstructed_matrix.at[indices].set(array)

    # Step 4: Fill the upper part of the matrix
    # Mirror the lower part to the upper part
    reconstructed_matrix = reconstructed_matrix + reconstructed_matrix.T - jnp.diag(jnp.diag(reconstructed_matrix))

    return reconstructed_matrix

def mu_estimator(z, mu, ceph, is_cal, H0, Omat, dM, no_dM=None):
    """Estimator for the luminosity distance, taken from `scoutpip`"""
    c = 299792.458  # speed of light in km/s

    # calculate dl_mod and mu_mod in a vectorized manner
    dl_mod = jnp.array([dl_monopole(z_i, H0, Omat) for z_i in z])
    mu_mod = 5 * jnp.log10(dl_mod) + 25
    
    delta_mu=jnp.zeros(1701)

    # determine delta_mu based on no_dM condition
    if no_dM is not None:
        delta_mu = mu - mu_mod
    else:
        delta_mu = jnp.where(is_cal == 1, mu + dM - ceph, mu + dM - mu_mod)

    H_array=Hz(z, H0, Omat)

    # numerator and denominator calculations
    num = c * jnp.log(10)
    den = 5 * (-1 + c * (1 + z)**2 / (dl_mod * H_array))

    num_den=num/den

    return num_den * delta_mu, num_den

def jax_integration(f, a, b, n, args):
    """
    Simple manual implementation of numerical integration with JAX using trapezoidal rule.
    """
    x = jnp.linspace(a, b, n)
    y = f(x, *args)
    dx = (b - a) / (n - 1)
    return jnp.sum((y[:-1] + y[1:]) * dx / 2)

def integrand(z, H0, Omat):
    return 1 / (H0 * jnp.sqrt(Omat * (1 + z) ** 3 + (1 - Omat)))

def dl_monopole(z, H0, Omat):
    """Simple integral of the monopole of the luminosity distance, in JAX"""
    c = 299792.458
    integral = jax_integration(integrand, 0.0, z, 1000, (H0, Omat))
    return c * (1 + z) * integral
    
