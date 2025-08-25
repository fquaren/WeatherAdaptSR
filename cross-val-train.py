import os
import random
import string
import torch
import yaml
import argparse
import pandas as pd
import optuna
import gc
import numpy as np
from data.dataloader import (
    get_single_cluster_dataloader,
    get_clusters_dataloader,
    get_domain_adaptation_dataloaders,
)
from src.models import unet
from src.train import train_model, objective
from src.train_mmd import train_model_mmd, objective_mmd
from src.logger import setup_logger


# Generate random experiment ID
def generate_experiment_id(length=4):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


# Launch experiment
def main():

    print("Starting cross-validation training ...")
    start_time = pd.Timestamp.now()
    print("Time: ", start_time)

    # Get argument for local or curnagl config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="curnagl", help="Local or curnagl config"
    )
    parser.add_argument(
        "--resume_exp", type=str, default=None, help="Local or curnagl config"
    )
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument(
        "--method",
        type=str,
        default="cross-val",
        help="Method name (cross-val, all, mmd)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Experiment seed, default 42."
    )
    args = parser.parse_args()
    config = "config_local" if args.config == "local" else "config_curnagl"
    resume_exp = args.resume_exp
    if args.model is None:
        print("INFO: No model specified. Using cross-val.")
        return
    model_name = args.model
    if args.method not in ["all", "cross-val", "mmd", "single"]:
        print(
            f"INFO: Method {args.method} not recognized. Please use 'cross-val', 'all', 'mmd', or 'single'."
        )
        return
    method = args.method

    # Experiment creation
    if resume_exp is None:
        # Load local config
        config_path = os.path.join(
            os.path.dirname(__file__), "configs", f"{config}.yaml"
        )
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        exp_id = generate_experiment_id()
        print(f"EXP: Generated new experiment ID: {exp_id}")
        exp_path = config["experiment"]["save_dir"]
        exp_name = config["experiment"]["name"]
        output_dir = os.path.join(exp_path, exp_name, exp_id)
        print(f"EXP: New experiment path: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        # Save config experiment
        with open(os.path.join(output_dir, "config.yaml"), "w") as file:
            yaml.dump(config, file)
            file.write(f"EXPERIMENT_ID: {exp_id}\n")
            file.write(f"EXPERIMENT_START_TIME: {start_time}\n")
            file.write(f"EXPERIMENT_MODEL: {model_name}\n")
    else:
        if os.path.isdir(resume_exp):
            print(
                f"EXP: Found experiment at {resume_exp} already exists. Resuming training."
            )
        else:
            print(f"EXP: Experiment {resume_exp} does not exist. Retry.")
            return
        output_dir = resume_exp
        print(f"EXP: Resuming experiment at: {output_dir}")
        exp_id = os.path.basename(os.path.normpath(output_dir))

    # Use saved config file for experiment
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Setup logger
    print(
        f"EXP: Setting up logger at {output_dir} with ID {exp_id} for model {model_name} using method {method}."
    )
    logger = setup_logger(output_dir, "experiment")
    logger.info(
        f"Starting cross-validation experiment (EXPERIMENT ID: {exp_id}, TIME: {start_time})"
    )
    logger.info(f"EXPERIMENT ID: {exp_id}")
    logger.info(f"EXPERIMENT OUTPUT DIRECTORY: {output_dir}")
    logger.info(f"EXPERIMENT START TIME: {start_time}")
    logger.info("---------------------------------")
    logger.info("EXPERIMENT CONFIGURATION:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Method: {method}")
    logger.info(f"  Resume: {resume_exp}")
    logger.info(f"  Config path: {config_path}")
    logger.info(f"  Config dump: \n{yaml.dump(config, sort_keys=False)}")
    logger.info("---------------------------------")

    # Set random seed for reproducibility
    set_seed(42)

    # Load data path and cluster names
    data_path = config["paths"]["data_path"]
    cluster_names = config["paths"]["clusters"]
    if cluster_names is None:
        cluster_names = sorted(
            [
                c
                for c in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, c))
            ]
        )

    # Device configuration
    if config["training"]["load_data_on_gpu"]:
        device_data = "cuda"
    else:
        device_data = "cpu"
    device = config["experiment"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("INFO: CUDA is not available. Using CPU instead.")
        device = "cpu"
    elif device == "cuda":
        print(f"INFO: Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("INFO: Using CPU.")

    if method == "single":

        clusters_to_process = cluster_names

        for cluster_to_process in clusters_to_process:
            logger.info(f"METHOD: 'single' will train on {cluster_to_process}.")
            single_cluster = True

            # Hyperparameter optimization
            if config["optimization"]["num_epochs"] != 0:
                logger.info(
                    f"OPTIMIZATION: Optimizing hyperparameters for {model_name}..."
                )
                num_epochs = config["optimization"]["num_epochs"]

                logger.info(f"MODEL: Loading model: {model_name} ...")
                model = getattr(unet, model_name)()
                if model is None:
                    logger.info(f"MODEL: Model {model_name} not found.")
                    return
                if torch.cuda.device_count() > 1:
                    logger.info(f"MODEL: Using {torch.cuda.device_count()} GPUs!")
                    model = torch.nn.DataParallel(model)
                else:
                    logger.info("MODEL: Using single GPU or CPU.")
                logger.info(f"MODEL: Moving model to device: {device} ...")
                model.apply(unet.init_weights_kaiming)
                model.to(device)

                logger.info(
                    f"OPTIMIZATION: Optimizing model for {cluster_to_process} ..."
                )
                study = optuna.create_study(direction="minimize")
                study.optimize(
                    lambda trial: objective(
                        trial,
                        model,
                        num_epochs,
                        cluster_to_process,
                        cluster_names,
                        config,
                        device,
                        device_data,
                        config["training"]["augmentation"],
                        single_cluster=single_cluster,
                    ),
                    n_trials=config["optimization"]["num_trials"],
                )

                if cluster_to_process in config["domain_specific"]:
                    ds = config["domain_specific"][cluster_to_process]
                    ds["optimizer_params"].update(
                        {
                            "lr_model": study.best_params["lr_model"],
                            "weight_decay": study.best_params["weight_decay"],
                        }
                    )
                    ds["loss_params"].update(
                        {
                            "lr_loss": study.best_params["lr_loss"],
                            "weight_decay": 0.0,
                        }
                    )
                with open(config_path, "w") as f:
                    yaml.dump(config, f, sort_keys=False)
            else:
                logger.info(
                    "OPTIMIZATION: Skipping hyperparameter optimization as num_epochs is 0."
                )

            # Reload config
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)

            # Training
            logger.info(f"TRAINING: Starting training for method: {method}")
            logger.info(f"MODEL: Loading model: {model_name} ...")
            model = getattr(unet, model_name)()
            if torch.cuda.device_count() > 1:
                logger.info(f"MODEL: Using {torch.cuda.device_count()} GPUs!")
                model = torch.nn.DataParallel(model)
            else:
                logger.info("MODEL: Using single GPU or CPU.")
            logger.info(f"MODEL: Moving model to device: {device} ...")
            model.apply(unet.init_weights_kaiming)
            model.to(device)

            logger.info(f"TRAINING: {cluster_to_process}")

            loaders = get_single_cluster_dataloader(
                data_path=config["paths"]["data_path"],
                elev_dir=config["paths"]["elev_path"],
                cluster_name=cluster_to_process,
                vars=config["experiment"]["vars"],
                batch_size=config["training"]["batch_size"],
                num_workers=config["training"]["num_workers"],
                use_theta_e=config["training"]["use_theta_e"],
                device=device_data,
                augment=config["training"]["augmentation"],
            )

            train_model(
                model=model,
                excluding_cluster=cluster_to_process,
                num_epochs=config["training"]["num_epochs"],
                train_loader=loaders["train"],
                val_loader=loaders["val"],
                config=config,
                device=device,
                save_path=output_dir,
            )

            logger.info(f"TRAINING: Finished training on cluster: {cluster_to_process}")
            logger.info(
                f"TRAINING: Emptying GPU memory for cluster: {cluster_to_process}"
            )

    if method == "all":

        logger.info("METHOD: 'all' will train on all clusters.")
        cluster_to_process = "all_clusters"
        single_cluster = False

        # Hyperparameter optimization
        if config["optimization"]["num_epochs"] != 0:
            logger.info(f"OPTIMIZATION: Optimizing hyperparameters for {model_name}...")
            num_epochs = config["optimization"]["num_epochs"]

            logger.info(f"MODEL: Loading model: {model_name} ...")
            model = getattr(unet, model_name)()
            if model is None:
                logger.info(f"MODEL: Model {model_name} not found.")
                return
            if torch.cuda.device_count() > 1:
                logger.info(f"MODEL: Using {torch.cuda.device_count()} GPUs!")
                model = torch.nn.DataParallel(model)
            else:
                logger.info("MODEL: Using single GPU or CPU.")
            logger.info(f"MODEL: Moving model to device: {device} ...")
            model.apply(unet.init_weights_kaiming)
            model.to(device)

            logger.info(f"OPTIMIZATION: Optimizing model for {cluster_to_process} ...")
            study = optuna.create_study(direction="minimize")
            study.optimize(
                lambda trial: objective(
                    trial,
                    model,
                    num_epochs,
                    cluster_to_process,
                    cluster_names,
                    config,
                    device,
                    device_data,
                    config["training"]["augmentation"],
                    single_cluster=single_cluster,
                ),
                n_trials=config["optimization"]["num_trials"],
            )

            if cluster_to_process in config["domain_specific"]:
                config["domain_specific"][cluster_to_process][
                    "optimizer_params"
                ].update(study.best_params)
            with open(config_path, "w") as f:
                yaml.dump(config, f, sort_keys=False)
        else:
            logger.info(
                "OPTIMIZATION: Skipping hyperparameter optimization as num_epochs is 0."
            )

        # Reload config
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Training
        logger.info(f"TRAINING: Starting training for method: {method}")
        logger.info(f"MODEL: Loading model: {model_name} ...")
        model = getattr(unet, model_name)()
        if torch.cuda.device_count() > 1:
            logger.info(f"MODEL: Using {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model)
        else:
            logger.info("MODEL: Using single GPU or CPU.")
        logger.info(f"MODEL: Moving model to device: {device} ...")
        model.apply(unet.init_weights_kaiming)
        model.to(device)

        logger.info(f"TRAINING: {cluster_to_process}")

        loaders = get_clusters_dataloader(
            data_path=config["paths"]["data_path"],
            elev_dir=config["paths"]["elev_path"],
            excluded_cluster=cluster_to_process,
            cluster_names=cluster_names,
            batch_size=config["training"]["batch_size"],
            num_workers=config["training"]["num_workers"],
            use_theta_e=config["training"]["use_theta_e"],
            device=device_data,
            augment=config["training"]["augmentation"],
        )

        train_model(
            model=model,
            excluding_cluster=cluster_to_process,
            num_epochs=config["training"]["num_epochs"],
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            config=config,
            device=device,
            save_path=output_dir,
        )

        logger.info(f"TRAINING: Finished training on cluster: {cluster_to_process}")
        logger.info(f"TRAINING: Emptying GPU memory for cluster: {cluster_to_process}")

    if method == "cross-val":

        logger.info(
            "METHOD: 'cross-val' will train on all clusters in a cross validation fashion."
        )
        clusters_to_process = cluster_names

        # Hyperparameter optimization
        if config["optimization"]["num_epochs"] != 0:
            logger.info(f"OPTIMIZATION: Optimizing hyperparameters for {model_name}...")
            num_epochs = config["optimization"]["num_epochs"]
            for cluster in clusters_to_process:
                logger.info(f"MODEL: Loading model: {model_name} ...")
                model = getattr(unet, model_name)()
                if model is None:
                    logger.info(f"MODEL: Model {model_name} not found.")
                    return
                if torch.cuda.device_count() > 1:
                    logger.info(f"MODEL: Using {torch.cuda.device_count()} GPUs!")
                    model = torch.nn.DataParallel(model)
                else:
                    logger.info("MODEL: Using single GPU or CPU.")
                logger.info(f"MODEL: Moving model to device: {device} ...")
                model.apply(unet.init_weights_kaiming)
                model.to(device)

                logger.info(f"OPTIMIZATION: Optimizing model excluding {cluster} ...")
                study = optuna.create_study(direction="minimize")
                study.optimize(
                    lambda trial: objective(
                        trial,
                        model,
                        num_epochs,
                        cluster,
                        cluster_names,
                        config,
                        device,
                        device_data,
                        config["training"]["augmentation"],
                    ),
                    n_trials=config["optimization"]["num_trials"],
                )

                if cluster in config["domain_specific"]:
                    config["domain_specific"][cluster]["optimizer_params"].update(
                        study.best_params
                    )
                with open(config_path, "w") as f:
                    yaml.dump(config, f, sort_keys=False)
        else:
            logger.info(
                "OPTIMIZATION: Skipping hyperparameter optimization as num_epochs is 0."
            )

        # Reload config
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Training
        logger.info(f"TRAINING: Starting training for method: {method}")
        for excluded_cluster in clusters_to_process:
            logger.info(f"MODEL: Loading model: {model_name} ...")
            model = getattr(unet, model_name)()
            if torch.cuda.device_count() > 1:
                logger.info(f"MODEL: Using {torch.cuda.device_count()} GPUs!")
                model = torch.nn.DataParallel(model)
            else:
                logger.info("MODEL: Using single GPU or CPU.")
            logger.info(f"MODEL: Moving model to device: {device} ...")
            model.apply(unet.init_weights_kaiming)
            model.to(device)

            logger.info(f"TRAINING: Excluding cluster: {excluded_cluster}")
            loaders = get_clusters_dataloader(
                data_path=config["paths"]["data_path"],
                elev_dir=config["paths"]["elev_path"],
                excluded_cluster=excluded_cluster,
                cluster_names=cluster_names,
                batch_size=config["training"]["batch_size"],
                num_workers=config["training"]["num_workers"],
                use_theta_e=config["training"]["use_theta_e"],
                device=device_data,
                augment=config["training"]["augmentation"],
            )

            train_model(
                model=model,
                excluding_cluster=excluded_cluster,
                num_epochs=config["training"]["num_epochs"],
                train_loader=loaders["train"],
                val_loader=loaders["val"],
                config=config,
                device=device,
                save_path=output_dir,
            )

            logger.info(
                f"TRAINING: Finished training excluding cluster: {excluded_cluster}"
            )
            logger.info(
                f"TRAINING: Emptying GPU memory for cluster: {excluded_cluster}"
            )
            for split in ["train", "val", "test"]:
                dataset = loaders[split].dataset
                for d in dataset.datasets:
                    d.unload_from_gpu()

            del model, loaders
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"TRAINING: GPU memory emptied for cluster: {excluded_cluster}")

    if method == "mmd":

        logger.info(f" *** TRAINING: Performing domain adaptation with {method}.")

        # Optimize hyperparameters
        if config["optimization"]["num_epochs"] != 0:
            logger.info(
                f" *** OPTIMIZATION: Optimizing hyperparameters for {model_name}..."
            )
            for cluster in cluster_names:
                # Load model on device (multi-GPU if available)
                logger.info(f"MODEL: Loading model: {model_name} ...")
                model = getattr(unet, model_name)()
                if model is None:
                    logger.info(
                        f"MODEL: Model {model_name} not found in unet module. Please check the model name."
                    )
                    return
                if torch.cuda.device_count() > 1:
                    logger.info(f"MODEL: Using {torch.cuda.device_count()} GPUs!")
                    model = torch.nn.DataParallel(model)  # Wrap model for multi-GPU
                else:
                    logger.info("MODEL: Using single GPU or CPU.")
                logger.info(f"MODEL: Moving model to device: {device} ...")
                model.apply(unet.init_weights_kaiming)
                model.to(device)

                logger.info(
                    f"OPTIMIZATION: Optimizing model excluding {cluster} for training ..."
                )
                num_epochs = config["optimization"]["num_epochs"]
                study = optuna.create_study(direction="minimize")
                study.optimize(
                    lambda trial: objective_mmd(
                        trial,
                        model,
                        num_epochs,
                        cluster,
                        cluster_names,
                        config,
                        device,
                        device_data,
                        config["training"]["augmentation"],
                    ),
                    n_trials=config["optimization"]["num_trials"],
                )

                # Update params
                if cluster in config["domain_specific"]:
                    config["domain_specific"][cluster]["optimizer_params"].update(
                        study.best_params
                    )
                with open(config_path, "w") as f:
                    yaml.dump(config, f, sort_keys=False)
            logger.info(" *** OPTIMIZATION: Done.")
        else:
            logger.info(
                " *** OPTIMIZATION: Skipping hyperparameter optimization as num_epochs is set to 0."
            )

        # Reload config
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Training
        logger.info(
            f" *** TRAINING: Training on all clusters in cross-validation fashion for {model_name}."
        )
        for excluded_cluster in cluster_names:

            # Load model on device (multi-GPU if available)
            logger.info(f"MODEL: Loading model: {model_name} ...")
            model = getattr(unet, model_name)()
            if torch.cuda.device_count() > 1:
                logger.info(f"MODEL: Using {torch.cuda.device_count()} GPUs!")
                model = torch.nn.DataParallel(model)  # Wrap model for multi-GPU
            else:
                logger.info("MODEL: Using single GPU or CPU.")
            logger.info(f"MODEL: Moving model to device: {device} ...")
            model.apply(unet.init_weights_kaiming)
            model.to(device)

            # Get dataloaders for all clusters excluding the current one
            logger.info(f"TRAINING: Excluding cluster: {excluded_cluster}")
            loaders = get_domain_adaptation_dataloaders(
                data_path=config["paths"]["data_path"],
                elev_dir=config["paths"]["elev_path"],
                target_cluster=excluded_cluster,
                cluster_names=cluster_names,
                batch_size=config["training"]["batch_size"],
                num_workers=config["training"]["num_workers"],
                use_theta_e=config["training"]["use_theta_e"],
                device=device_data,
                augment=config["training"]["augmentation"],
            )

            train_model_mmd(
                model=model,
                excluding_cluster=excluded_cluster,
                num_epochs=config["training"]["num_epochs"],
                source_train_loader=loaders["source"]["train"],
                target_train_loader=loaders["target"]["train"],
                source_val_loader=loaders["source"]["val"],
                config=config,
                device=device,
                save_path=output_dir,
            )

            # Empty gpu memory
            logger.info(
                f"TRAINING: Finished training excluding cluster: {excluded_cluster}"
            )
            logger.info(
                f"TRAINING: Emptying GPU memory for cluster: {excluded_cluster}"
            )
            for split in ["train", "val", "test"]:
                source_ds = loaders["source"][split].dataset
                for d in source_ds.datasets:
                    d.unload_from_gpu()
                target_ds = loaders["target"][split].dataset
                target_ds.unload_from_gpu()
            del model, loaders
            # Empty GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"TRAINING: GPU memory emptied for cluster: {excluded_cluster}")

    logger.info("(**************************************************************) ")
    logger.info("(***) TRAINING: All clusters trained. Experiment completed.(***) ")
    logger.info("(**************************************************************) ")
    logger.info(f"TRAINING: Experiment saved at: {output_dir}")
    logger.info(f"TRAINING: Elapsed time: {pd.Timestamp.now() - start_time}")

    # Save config experiment
    with open(os.path.join(output_dir, "config.yaml"), "w") as file:
        yaml.dump(config, file)
        file.write(f"EXPERIMENT_ELAPSED_TIME: {pd.Timestamp.now() - start_time}\n")

    print(f"Experiment completed. Output directory: {output_dir}")

    return


if __name__ == "__main__":
    main()
