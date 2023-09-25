from jax.random import PRNGKey

from bsn_jax.data.data import GenzContinuousDataSet1D
from bsn_jax.options import Options, MethodEnum
from bsn_jax.run_model import run

if __name__ == "__main__":
    prng_key = PRNGKey(0)
    opts = Options(
        step_size=0.01,
        method=MethodEnum.L_BFGS_B,
        num_epochs=1000,
        layer_sizes=[[1, 32], [32, 1]],
        n=10,
        data_class=GenzContinuousDataSet1D,
    )

    run(opts, prng_key)
