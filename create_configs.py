import toml

def create_accelerate_config(file_path):
    config = {
        "compute_environment": {
            "distributed_type": "NO",
            "mixed_precision": "no"
        },
        "resources": {
            "cpu": True,
            "num_cpus": 1
        }
    }
    with open(file_path, 'w') as f:
        toml.dump(config, f)
    print(f"Accelerate config saved to {file_path}")

def create_dataset_config(file_path):
    config = {
        "datasets": [
            {
                "subsets": [
                    {
                        "num_repeats": 10,
                        "image_dir": "images"
                    }
                ]
            }
        ],
        "general": {
            "resolution": 512,
            "shuffle_caption": True,
            "keep_tokens": 1,
            "flip_aug": True,
            "caption_extension": ".txt",
            "enable_bucket": True,
            "bucket_reso_steps": 64,
            "bucket_no_upscale": False,
            "min_bucket_reso": 320,
            "max_bucket_reso": 1024
        }
    }
    with open(file_path, 'w') as f:
        toml.dump(config, f)
    print(f"Dataset config saved to {file_path}")

def create_training_config(file_path):
    config = {
        "additional_network_arguments": {
            "unet_lr": 0.0005,
            "text_encoder_lr": 0.0001,
            "network_dim": 16,
            "network_alpha": 8,
            "network_module": "networks.lora",
            "network_train_unet_only": True
        },
        "optimizer_arguments": {
            "learning_rate": 0.0005,
            "lr_scheduler": "cosine_with_restarts",
            "lr_scheduler_num_cycles": 3,
            "lr_warmup_steps": 500,
            "optimizer_type": "AdamW"
        },
        "training_arguments": {
            "max_train_epochs": 10,
            "save_every_n_epochs": 1,
            "save_last_n_epochs": 10,
            "train_batch_size": 2,
            "output_dir": "output",
            "logging_dir": "logs"
        },
        "model_arguments": {
            "pretrained_model_name_or_path": "sd-v1-5-pruned-noema-fp16.safetensors"
        }
    }
    with open(file_path, 'w') as f:
        toml.dump(config, f)
    print(f"Training config saved to {file_path}")

# Define the paths for the TOML files
dataset_config_file = "configs/dataset_config.toml"
config_file = "configs/training_config.toml"

# Create the TOML files
create_dataset_config(dataset_config_file)
create_training_config(config_file)
