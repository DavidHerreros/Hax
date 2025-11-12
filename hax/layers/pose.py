import jax
import jax.numpy as jnp
import jax.random as random


def rot6d_to_rotmat(x6):
    """
    Gram-Schmidt process to recover a matrix from a 6D vector representation
    """
    a1 = x6[..., 0:3]
    a2 = x6[..., 3:6]

    b1 = a1 / (jnp.linalg.norm(a1, axis=-1, keepdims=True) + 1e-9)
    a2_proj = a2 - jnp.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2_proj / (jnp.linalg.norm(a2_proj, axis=-1, keepdims=True) + 1e-9)
    b3 = jnp.cross(b1, b2)
    R = jnp.stack([b1, b2, b3], axis=-1)
    return R

def so3_hat(v):
    """ Convert 3D vector to skew-symmetric matrix """
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    O = jnp.zeros_like(vx)
    return jnp.stack([
        jnp.stack([ O, -vz,  vy], axis=-1),
        jnp.stack([ vz,  O, -vx], axis=-1),
        jnp.stack([-vy,  vx,  O], axis=-1),
    ], axis=-2)  # (..., 3, 3)

def so3_expm(omega):
    """
    Based on Rodrigues’ formula, computes a tiny rotation ΔR
    """
    theta = jnp.linalg.norm(omega, axis=-1, keepdims=True)
    K = so3_hat(omega / (theta + 1e-9))
    I = jnp.eye(3, dtype=omega.dtype)

    th = theta[..., 0]
    A = jnp.where(th < 1e-4, 1.0 - th**2/6.0, jnp.sin(th)/th)
    B = jnp.where(th < 1e-4, 0.5 - th**2/24.0, (1.0 - jnp.cos(th))/(th**2 + 1e-12))

    A = A[..., None, None]
    B = B[..., None, None]
    return I + A * K + B * (K @ K)

def sample_topM_R(rng, R_mode, log_scale, M):
    """
    Sampling of the top M rotation matrices based on a rotation mode
    """
    Rs, omegas, log_q = PoseDistMatrix.sample(rng, R_mode, log_scale, M)
    return Rs, omegas, log_q

def importance_weights(neg_log_likes, log_q):
    """
    For Mixture of Experts: finds the importance weight of each sample based on its loss
    """
    log_w = -neg_log_likes - log_q
    log_w = log_w - jax.nn.logsumexp(log_w, axis=-1, keepdims=True)
    return jnp.exp(log_w), log_w

class PoseDistMatrix:
    """
    Static class to easily handle rotation operations
    """

    @staticmethod
    def mode_rotmat(x6_mode):
        """
        Wrapper around method to get valid rotation  matrices from 6D vector representations.
        Both, batched (B,M,M) or unbatched (M,M) rotations are allowed
        """
        return rot6d_to_rotmat(x6_mode)

    @staticmethod
    def sample(rng, R_mode, log_scale, M: int):
        """
        Utility to sample rotations around a given rotation mode.
        Both, batched (B,M,M) or unbatched (M,M) rotations are allowed
        """
        def single(Rm, ls, key_eps):
            # key_eps = key_eps() if callable(key_eps) else key_eps
            eps = random.normal(key_eps, (M, 3))
            sigma = jnp.exp(ls)
            omega = eps * sigma
            dR = jax.vmap(so3_expm)(omega)
            R = jax.vmap(lambda A: A @ Rm)(dR)  # World frame rotation
            # R = jax.vmap(lambda A: Rm @ A)(dR)  # Camera/local-frame rotation
            log_q = PoseDistMatrix.log_prob(omega, ls)
            return R, omega, log_q

        if R_mode.ndim == 2:
            key, _ = random.split(rng, 2)
            return single(R_mode, log_scale, key)
        else:
            B = R_mode.shape[0]
            keys = random.split(rng, B)
            def body(i):
                return single(R_mode[i], log_scale[i], keys[i])
            Rs, omegas, lq = jax.vmap(body)(jnp.arange(B))
            return Rs, omegas, lq

    @staticmethod
    def log_prob(omega, log_scale):
        var = jnp.exp(2.0 * log_scale)
        quad = jnp.sum((omega ** 2.) / (var + 1e-9), axis=-1)
        log_det = jnp.sum(2.0 * log_scale, axis=-1)
        return -0.5 * (quad + 3. * jnp.log(2. * jnp.pi) + log_det)

    @staticmethod
    def kl_to_isotropic_prior(log_scale, prior_log_scale=0.0):
        s2 = jnp.exp(2. * log_scale)
        p2 = jnp.exp(2. * prior_log_scale)
        term = (s2 / p2) - 1. - jnp.log(s2/p2 + 1e-9)
        return 0.5 * jnp.sum(term, axis=-1)
