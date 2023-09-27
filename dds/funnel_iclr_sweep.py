# %%
from dds.configs.config import set_task
from dds.train_dds import train_dds


import numpy as onp
from dds.utils import flatten_nested_dict, update_config_dict
import jax
from absl import app, flags
from ml_collections import config_dict as configdict
from ml_collections import config_flags

import wandb

FLAGS = flags.FLAGS

opt_funnel = {
    'funnel': {
        '32': {
            'oudstl': {
                'sigma': 1.075,
                'alpha': 1.075,
                'm': None
            },
            'pisstl': {
                'sigma': 1.0675000000000001,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 1.85,
                'alpha': 1.67,
                'm': 0.9
            }
        },
        '64': {
            'oudstl': {
                'sigma': 1.075,
                'alpha': 0.6875,
                'm': None
            },
            'pisstl': {
                'sigma': 0.4158333333333334,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 1.85,
                'alpha': 3.7,
                'm': 0.9
            }
        },
        '128': {
            'oudstl': {
                'sigma': 1.85,
                'alpha': 0.3,
                'm': None
            },
            'pisstl': {
                'sigma': 0.7416666666666667,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 1.075,
                'alpha': 2.5,
                'm': 0.9
            }
        },
        '256': {
            'oudstl': {
                'sigma': 1.4625000000000001,
                'alpha': 0.3,
                'm': None
            },
            'pisstl': {
                'sigma': 0.2529166666666667,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 0.6875,
                'alpha': 3.7,
                'm': 0.9
            }
        }
    }
}


def main(funnel_config):

    # funnel_config = get_config()


    wandb_kwargs = {
            "project": funnel_config.wandb.project,
            "entity": funnel_config.wandb.entity,
            "config": flatten_nested_dict(funnel_config.to_dict()),
            "mode": "online" if funnel_config.wandb.log else "disabled",
            # "settings": wandb.Settings(code_dir=funnel_config.wandb.code_dir),
        }

    with wandb.init(**wandb_kwargs) as run:
    # %%
        update_config_dict(funnel_config, run, {})
        # Time and step settings (Need to be done before calling set_task)
        # funnel_config.model.tfinal = 1.6
        funnel_config.model.dt = 0.05

        if funnel_config.model.reference_process_key == "oudstl":
            funnel_config.model.step_scheme_key = "cos_sq"

        funnel_config = set_task(funnel_config, "funnel")
        funnel_config.model.reference_process_key = "oudstl"

        if funnel_config.model.reference_process_key == "oudstl":
            funnel_config.model.step_scheme_key = "cos_sq"
            
            # Opt setting for funnel
            key = int(funnel_config.model.tfinal / funnel_config.model.dt)
            key = 32 if key < 32 else key

            print(f'K: {key}')
            # funnel_config.model.sigma = 1.075
            # funnel_config.model.alpha = 0.6875
            # funnel_config.model.m = 1.0

            funnel_config.model.sigma = opt_funnel['funnel'][str(key)][funnel_config.model.reference_process_key]['sigma']
            funnel_config.model.alpha = opt_funnel['funnel'][str(key)][funnel_config.model.reference_process_key]['alpha']
            funnel_config.model.m = opt_funnel['funnel'][str(key)][funnel_config.model.reference_process_key]['m']
                
            # Path opt settings    
            funnel_config.model.exp_dds = False


        funnel_config.model.stl = False
        funnel_config.model.detach_stl_drift = False

        funnel_config.trainer.notebook = True
        # Opt settings we use
        # funnel_config.trainer.learning_rate = 0.0001
        funnel_config.trainer.learning_rate = 5 * 10**(-3)
        funnel_config.trainer.lr_sch_base_dec = 0.95 # For funnel

        print(funnel_config)
        # %%
        # funnel_config.trainer.epochs = 2000
        out_dict = train_dds(funnel_config, run)

        # %%
        out_dict[-1].keys()

        # %%
        onp.mean(out_dict[-1]["is_eval"])

        # %%
        onp.mean(out_dict[-1]["pf_eval"])

        # %%
        out_dict[-1]["pf_eval"]

        # %%
        funnel_config.model.reference_process_key
        from dds.targets.toy_targets import funnel

        import ot
        def W2_distance(x, y, reg = 0.01):
            N = x.shape[0]
            x, y = onp.array(x), onp.array(y)
            a,b = onp.ones(N) / N, onp.ones(N) / N

            M = ot.dist(x, y)
            M /= M.max()

            T_reg = ot.sinkhorn2(
                a, b, M, reg, log=False,
                numItermax=10000, stopThr=1e-16
            )
            return T_reg



        print({
            'final_ln_Z': -onp.mean(out_dict[-1]["is_eval"]),
            'final_ln_Z_std': onp.std(out_dict[-1]["is_eval"]),
        })

        run.log({
            'final_ln_Z': -onp.mean(out_dict[-1]["is_eval"]),
            'final_ln_Z_std': onp.std(out_dict[-1]["is_eval"]),
        })

        params, model_state, forward_fn_wrap, rng_key, results_dict = out_dict
        print(results_dict.keys())
        neg_energy, sample = funnel()


        (augmented_trajectory, _), _ = forward_fn_wrap(params, model_state, rng_key,
                                                        15000)

        n_seeds = 30
        samples = augmented_trajectory[:, -1, :10]

        target_samples = sample(15000)

        print(samples.shape, target_samples.shape)

        num_samples_per_seed = 500

        w2_dists = []

        for i in range(n_seeds):
            target_samples_i = target_samples[i * num_samples_per_seed: (i + 1) * num_samples_per_seed]
            samples_i = samples[i * num_samples_per_seed: (i + 1) * num_samples_per_seed]
            w2_dists.append(W2_distance(target_samples_i, samples_i))

        print(onp.mean(w2_dists), onp.std(w2_dists))

        run.log({
            'W2': onp.mean(w2_dists),
            'W2_std': onp.std(w2_dists),
        })
        


if __name__ == "__main__":
    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)