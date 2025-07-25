import datetime
import logging
import os
from pathlib import Path

from omegaconf import OmegaConf

from src.configs import default_config
from src.utils import fix_random_seeds, get_sha, initialize_wandb, setup_logging

logger = logging.getLogger("ab-mil")


def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_args(args):
    default_cfg = OmegaConf.create(default_config)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))
    OmegaConf.resolve(cfg)
    return cfg


def setup(args):
    """
    Basic configuration setup without any distributed or GPU-specific initialization.
    This function:
      - Loads the config from file and command-line options.
      - Sets up logging.
      - Fixes random seeds.
      - Creates the output directory.
    """
    cfg = get_cfg_from_args(args)

    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")

    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("processed", summary="max")
        run_id = wandb_run.id

    output_dir = Path(cfg.output_dir, run_id)
    output_dir.mkdir(exist_ok=True, parents=True)
    cfg.output_dir = str(output_dir)

    fix_random_seeds(0)
    setup_logging(output=cfg.output_dir, level=logging.INFO)
    logger.info("git:\n  {}\n".format(get_sha()))
    cfg_path = write_config(cfg, cfg.output_dir)
    if cfg.wandb.enable:
        wandb_run.save(cfg_path)
    return cfg
