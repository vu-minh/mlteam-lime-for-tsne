# config params to run experiments with the selected points

config_selected_points = dict(
    # COUNTRY dataset
    country1=dict(
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
            "5": dict(
                name="Canada",
                reject_radius=5,
                use_weights=False,
                lambda_params=dict(
                    lower_bound_lambda=0.0001, upper_bound_lambda=3.5, nb_lambda=20,
                ),
            ),
            "9": dict(
                name="Netherlands",
                reject_radius=5,
                use_weights=False,
                lambda_params=dict(
                    lower_bound_lambda=0.0001, upper_bound_lambda=3.5, nb_lambda=20,
                ),
            ),
            "14": dict(
                name="Denmark",
                reject_radius=5,
                use_weights=False,
                lambda_params=dict(
                    lower_bound_lambda=0.0001, upper_bound_lambda=3.5, nb_lambda=20,
                ),
            ),
            "96": dict(
                name="South Africa",
                reject_radius=5,
                use_weights=False,
                lambda_params=dict(
                    lower_bound_lambda=0.0001, upper_bound_lambda=3.5, nb_lambda=20,
                ),
            ),
            "81": dict(
                name="Algeria",
                reject_radius=5,
                use_weights=False,
                lambda_params=dict(
                    lower_bound_lambda=0.0001, upper_bound_lambda=3.5, nb_lambda=20,
                ),
            ),
            "40": dict(
                name="Croatia",
                reject_radius=5,
                use_weights=False,
                lambda_params=dict(
                    lower_bound_lambda=0.0001, upper_bound_lambda=3.5, nb_lambda=20,
                ),
            ),
            "23": dict(
                name="Singapore",
                reject_radius=5,
                use_weights=False,
                lambda_params=dict(
                    lower_bound_lambda=0.0001, upper_bound_lambda=3.5, nb_lambda=20,
                ),
            ),
            # "0": dict(
            #     name="Norway",
            #     reject_radius=5,
            #     use_weights=False,
            #     lambda_params=dict(
            #         lower_bound_lambda=0.0001, upper_bound_lambda=5, nb_lambda=10,
            #     ),
            # ),
            # "4": dict(
            #     name="Sweden",
            #     reject_radius=5,
            #     use_weights=False,
            #     lambda_params=dict(
            #         lower_bound_lambda=0.0001, upper_bound_lambda=5, nb_lambda=10,
            #     ),
            # ),
            # "6": dict(
            #     name="Japan",
            #     reject_radius=5,
            #     use_weights=False,
            #     lambda_params=dict(
            #         lower_bound_lambda=0.0001, upper_bound_lambda=3.5, nb_lambda=20,
            #     ),
            # ),
            # "125": dict(
            #     name="Benin",
            #     reject_radius=5,
            #     use_weights=False,
            #     lambda_params=dict(
            #         lower_bound_lambda=0.0001, upper_bound_lambda=3.5, nb_lambda=20,
            #     ),
            # ),
            # "55": dict(
            #     name="Brazil",
            #     reject_radius=5,
            #     use_weights=False,
            #     lambda_params=dict(
            #         lower_bound_lambda=0.0001, upper_bound_lambda=3.5, nb_lambda=20,
            #     ),
            # ),
            # "106": dict(
            #     name="Bangladesh",
            #     reject_radius=5,
            #     use_weights=False,
            #     lambda_params=dict(
            #         lower_bound_lambda=0.0001, upper_bound_lambda=3.5, nb_lambda=20,
            #     ),
            # ),
            # "89": dict(
            #     name="Egypt",
            #     reject_radius=5,
            #     use_weights=False,
            #     lambda_params=dict(
            #         lower_bound_lambda=0.0001, upper_bound_lambda=3.5, nb_lambda=20,
            #     ),
            # ),
            # "7": dict(
            #     name="United States",
            #     reject_radius=5,
            #     use_weights=False,
            #     lambda_params=dict(
            #         lower_bound_lambda=0.0001, upper_bound_lambda=3.5, nb_lambda=20,
            #     ),
            # ),
            # "17": dict(
            #     name="United Kingdom",
            #     reject_radius=5,
            #     use_weights=False,
            #     lambda_params=dict(
            #         lower_bound_lambda=0.0001, upper_bound_lambda=3.5, nb_lambda=20,
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
