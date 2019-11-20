# config params to run experiments with the selected points

config_selected_points = dict(
    # COUNTRY dataset
    country=dict(
        # sampling params
        n_samples=100,
        seed=42,
        # basic params to run tsne the first time
        tsne_hyper_params=dict(method="exact", perplexity=10, n_iter=1000, random_state=42,),
        # params to re-run tsne quickly
        early_stop_hyper_params=dict(
            early_exaggeration=1, n_iter_without_progress=100, min_grad_norm=1e-7, n_iter=500,
        ),
        # selected points
        selected_points={
            "12": dict(
                name="Belgium",
                reject_radius=5,
                use_weights=False,
                lambda_params=dict(
                    lower_bound_lambda=0.01, upper_bound_lambda=1.0, nb_lambda=10,
                ),
            ),
            "6": dict(
                name="Japan",
                reject_radius=5,
                use_weights=False,
                lambda_params=dict(
                    lower_bound_lambda=0.01, upper_bound_lambda=1.0, nb_lambda=10,
                ),
            ),
            "125": dict(
                name="Benin",
                reject_radius=5,
                use_weights=False,
                lambda_params=dict(
                    lower_bound_lambda=0.01, upper_bound_lambda=1.0, nb_lambda=10,
                ),
            ),
            "55": dict(
                name="Brazil",
                reject_radius=5,
                use_weights=False,
                lambda_params=dict(
                    lower_bound_lambda=0.01, upper_bound_lambda=1.0, nb_lambda=10,
                ),
            ),
            "106": dict(
                name="Bangladesh",
                reject_radius=5,
                use_weights=False,
                lambda_params=dict(
                    lower_bound_lambda=0.01, upper_bound_lambda=1.0, nb_lambda=10,
                ),
            ),
        },
    ),
    # WINE dataset
    wine=dict(),
)
