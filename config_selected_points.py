# config params to run experiments with the selected points

config_selected_points = dict(
    # COUNTRY dataset
    country=dict(
        # sampling params
        n_samples=100,
        n_neighbors_SMOTE=10,
        seed=42,
        # basic params to run tsne the first time
        tsne_hyper_params=dict(method="exact", perplexity=10, n_iter=1000, random_state=42,),
        # params to re-run tsne quickly
        early_stop_hyper_params=dict(
            early_exaggeration=1, n_iter_without_progress=100, min_grad_norm=1e-7, n_iter=500,
        ),
        # selected points
        selected_points={
            # "97": dict(
            #     name="Morocco",
            #     reject_radius=5,
            #     use_weights=False,
            #     lambda_params=dict(
            #         lower_bound_lambda=0.01, upper_bound_lambda=10, nb_lambda=20,
            #     ),
            # ),
            # "127": dict(
            #     name="Zambia",
            #     reject_radius=5,
            #     use_weights=False,
            #     lambda_params=dict(
            #         lower_bound_lambda=0.01, upper_bound_lambda=10, nb_lambda=20,
            #     ),
            # ),
            "70": dict(
                name="Tunisia",
                reject_radius=5,
                use_weights=False,
                lambda_params=dict(
                    lower_bound_lambda=0.01, upper_bound_lambda=10, nb_lambda=20,
                ),
            ),
            "45": dict(
                name="Bulgaria",
                reject_radius=5,
                use_weights=False,
                lambda_params=dict(
                    lower_bound_lambda=0.01, upper_bound_lambda=10, nb_lambda=20,
                ),
            ),
            # "40": dict(
            #     name="Croatia",
            #     reject_radius=5,
            #     use_weights=False,
            #     lambda_params=dict(
            #         lower_bound_lambda=0.01, upper_bound_lambda=10, nb_lambda=20,
            #     ),
            # ),
            # "37": dict(
            #     name="Lithuania",
            #     reject_radius=5,
            #     use_weights=False,
            #     lambda_params=dict(
            #         lower_bound_lambda=0.01, upper_bound_lambda=10, nb_lambda=20,
            #     ),
            # ),
            "18": dict(
                name="Spain",
                reject_radius=5,
                use_weights=False,
                lambda_params=dict(
                    lower_bound_lambda=0.01, upper_bound_lambda=10, nb_lambda=20,
                ),
            ),
            # "14": dict(
            #     name="Denmark",
            #     reject_radius=5,
            #     use_weights=False,
            #     lambda_params=dict(
            #         lower_bound_lambda=0.01, upper_bound_lambda=10, nb_lambda=20,
            #     ),
            # ),
        },
        bad_points={
            "12": dict(
                name="Belgium",
                reject_radius=5,
                use_weights=False,
                lambda_params=dict(
                    lower_bound_lambda=0.0001, upper_bound_lambda=10, nb_lambda=10,
                ),
            ),
        },
    ),
    # WINE dataset
    wine=dict(),
)
