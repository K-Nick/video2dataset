"""Create dataset from video links and metadata."""

import os
import sys
import signal
import fire
import fsspec
from argparse import ArgumentParser

from omegaconf import OmegaConf
from kn_util.config import LazyConfig
from typing import List, Optional, Any
import numpy as np  # pylint: disable=unused-import

from video2dataset.logger import LoggerProcess
from video2dataset.data_writer import (
    WebDatasetSampleWriter,
    FilesSampleWriter,
    ParquetSampleWriter,
    TFRecordSampleWriter,
    DummySampleWriter,
)
from video2dataset.input_sharder import InputSharder
from video2dataset.output_sharder import OutputSharder
from video2dataset.distributor import (
    no_distributor,
    multiprocessing_distributor,
    pyspark_distributor,
    SlurmDistributor,
    SlurmShardSampler,
)
from video2dataset.workers import DownloadWorker, SubsetWorker, OpticalFlowWorker, CaptionWorker, WhisperWorker
from video2dataset.configs import CONFIGS
from video2dataset.types import EncodeFormats


def identity(x):
    return x


def get_args():

    parser = ArgumentParser("Create datasets from video/audio links")

    parser.add_argument(
        "--url_list", required=True, help="List of input URLs in supported input formats (csv, parquet, braceexpand tar paths, etc.)"
    )
    parser.add_argument("--output_folder", default="dataset", help="Desired location of output dataset")
    parser.add_argument(
        "--output_format", default="files", choices=["files", "webdataset", "parquet", "tfrecord", "dummy"], help="Format of output dataset"
    )
    parser.add_argument(
        "--input_format",
        default="csv",
        choices=["txt", "csv", "tsv", "tsv.gz", "json", "parquet", "webdataset"],
        help="Format of the input",
    )
    parser.add_argument(
        "--encode_formats",
        type=eval,
        help='Dict specifying what extension each modality should use, e.g., \'{"video": "mp4", "audio": "m4a"}\'',
    )
    parser.add_argument("--stage", default="download", help="Processing stage (download, subset, optical_flow, caption)")
    parser.add_argument("--url_col", default="url", help="Column in input containing the URL")
    parser.add_argument("--caption_col", help="(Deprecated!) Column in input containing captions (to be written as txt)")
    parser.add_argument("--clip_col", help="(Deprecated!) Column in input containing timeframes of clips for how to split video")
    parser.add_argument("--save_additional_columns", nargs="+", help="List of column names to save to json component of a sample")
    parser.add_argument("--enable_wandb", action="store_true", help="Enable logging info to wandb")
    parser.add_argument("--wandb_project", default="video2dataset", help="Name of wandb project to log runs to")
    parser.add_argument("--incremental_mode", default="incremental", choices=["incremental", "overwrite"], help="How to handle restarting")
    parser.add_argument("--max_shard_retry", type=int, default=1, help="Maximum attempts to retry a failed shard")
    parser.add_argument("--tmp_dir", default="/tmp", help="Path to temporary directory on your file system")
    parser.add_argument("--config", default="default", help="Path to your config of choice or the config itself")
    parser.add_argument("--opts", nargs="+", help="Additional config options to override", default=[])
    parser.add_argument("--delimiter", default=",", help="Delimiter for csv/tsv files")
    parser.add_argument("--key_col", default=None, help="Column in input containing the key for the sample")

    return parser.parse_args()


# pylint: disable=unused-argument
# pylint: disable=eval-used
# pylint: disable=broad-except
def video2dataset(local_args):

    clip_col = local_args.clip_col
    caption_col = local_args.caption_col
    encode_formats = local_args.encode_formats
    enable_wandb = local_args.enable_wandb
    incremental_mode = local_args.incremental_mode
    max_shard_retry = local_args.max_shard_retry
    output_folder = local_args.output_folder
    output_format = local_args.output_format
    save_additional_columns = local_args.save_additional_columns
    stage = local_args.stage
    url_col = local_args.url_col
    url_list = local_args.url_list
    wandb_project = local_args.wandb_project
    tmp_dir = local_args.tmp_dir
    config = local_args.config
    input_format = local_args.input_format
    opts = local_args.opts

    if isinstance(config, str):
        config = CONFIGS[config] if config in CONFIGS else OmegaConf.load(config)
        LazyConfig.apply_overrides(config, opts)
        config = OmegaConf.to_container(config)
    for arg_type in ["subsampling", "reading", "storage", "distribution"]:
        assert arg_type in config

    if config["reading"]["sampler"] is None:
        config["reading"]["sampler"] = identity

    called_from_slurm = "CALLED_FROM_SLURM" in os.environ
    if called_from_slurm:
        global_task_id = int(os.environ["GLOBAL_RANK"])
        num_tasks = config["distribution"]["distributor_args"]["n_nodes"] * config["distribution"]["distributor_args"]["tasks_per_node"]
        config["reading"]["sampler"] = SlurmShardSampler(global_task_id=global_task_id, num_tasks=num_tasks)
        config["distribution"]["distributor"] = "multiprocessing"

        # Only log from master
        enable_wandb = enable_wandb and (global_task_id == 0)

    # TODO: find better location for this code
    # TODO: figure out minimum yt_meta_args for subtitles to be added to metadata
    if config["storage"]["captions_are_subtitles"]:
        assert clip_col is None  # no weird double-clipping
        if config["reading"]["yt_args"]["yt_metadata_args"] is None:
            config["reading"]["yt_args"]["yt_metadata_args"] = {}
        if not config["reading"]["yt_args"]["yt_metadata_args"].get("writesubtitles", None):  # type: ignore
            config["reading"]["yt_args"]["yt_metadata_args"]["writesubtitles"] = "all"  # type: ignore

    if encode_formats is None:
        encode_formats = {"video": "mp4"}

    def make_path_absolute(path):
        fs, p = fsspec.core.url_to_fs(path)
        if fs.protocol == "file":
            return os.path.abspath(p)
        return path

    output_folder = make_path_absolute(output_folder)
    url_list = make_path_absolute(url_list)

    logger_process = LoggerProcess(output_folder, enable_wandb, wandb_project, local_args)
    tmp_path = output_folder + "/_tmp"
    fs, run_tmp_dir = fsspec.core.url_to_fs(tmp_path)
    if not fs.exists(run_tmp_dir):
        fs.mkdir(run_tmp_dir)

    def signal_handler(signal_arg, frame):  # pylint: disable=unused-argument
        try:
            fs.rm(run_tmp_dir, recursive=True)
        except Exception as _:  # pylint: disable=broad-except
            pass
        logger_process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    save_caption = caption_col is not None or config["storage"]["captions_are_subtitles"]

    fs, output_path = fsspec.core.url_to_fs(output_folder)

    if not fs.exists(output_path):
        fs.mkdir(output_path)
        done_shards = set()
    else:
        if incremental_mode == "incremental":
            done_shards = set(int(x.split("/")[-1].split("_")[0]) for x in fs.glob(output_path + "/*.json"))
        elif incremental_mode == "overwrite":
            fs.rm(output_path, recursive=True)
            fs.mkdir(output_path)
            done_shards = set()
        else:
            raise ValueError(f"Unknown incremental mode {incremental_mode}")

    logger_process.done_shards = done_shards
    logger_process.start()

    if output_format == "webdataset":
        sample_writer_class = WebDatasetSampleWriter
    elif output_format == "parquet":
        sample_writer_class = ParquetSampleWriter  # type: ignore
    elif output_format == "files":
        sample_writer_class = FilesSampleWriter  # type: ignore
    elif output_format == "tfrecord":
        sample_writer_class = TFRecordSampleWriter  # type: ignore
    elif output_format == "dummy":
        sample_writer_class = DummySampleWriter  # type: ignore
    else:
        raise ValueError(f"Invalid output format {output_format}")

    if input_format == "webdataset":
        shard_iterator = OutputSharder(url_list, input_format, done_shards, sampler=config["reading"]["sampler"])  # type: ignore
    else:
        shard_iterator = InputSharder(  # type: ignore
            url_list,
            input_format,
            url_col,
            caption_col,
            clip_col,
            save_additional_columns,
            config["storage"]["number_sample_per_shard"],
            done_shards,
            tmp_path,
            config["reading"]["sampler"],
            delimiter=local_args.delimiter,
        )

    if stage == "download":
        worker = DownloadWorker(
            sample_writer_class=sample_writer_class,
            save_caption=save_caption,
            output_folder=output_folder,
            column_list=shard_iterator.column_list,
            tmp_dir=tmp_dir,
            encode_formats=encode_formats,
            config=config,
            key_col=local_args.key_col,
        )
    elif stage == "subset":
        worker = SubsetWorker(  # type: ignore
            sample_writer_class=sample_writer_class,
            output_folder=output_folder,
            encode_formats=encode_formats,
            config=config,
        )
    elif stage == "optical_flow":
        is_slurm_task = "GLOBAL_RANK" in os.environ and config["distribution"]["distributor"] == "multiprocessing"
        worker = OpticalFlowWorker(  # type: ignore
            sample_writer_class=sample_writer_class,
            output_folder=output_folder,
            encode_formats=encode_formats,
            is_slurm_task=is_slurm_task,
            config=config,
        )
    elif stage == "caption":
        is_slurm_task = "GLOBAL_RANK" in os.environ and config["distribution"]["distributor"] == "multiprocessing"
        worker = CaptionWorker(  # type: ignore
            sample_writer_class=sample_writer_class,
            output_folder=output_folder,
            encode_formats=encode_formats,
            is_slurm_task=is_slurm_task,
            config=config,
        )
    elif stage == "whisper":
        is_slurm_task = "GLOBAL_RANK" in os.environ and config["distribution"]["distributor"] == "multiprocessing"
        worker = WhisperWorker(  # type: ignore
            sample_writer_class=sample_writer_class,
            output_folder=output_folder,
            column_list=shard_iterator.column_list,
            tmp_dir=tmp_dir,
            encode_formats=encode_formats,
            is_slurm_task=is_slurm_task,
            config=config,
        )
    else:
        raise ValueError(f"Invalid stage: {stage}")

    print("Starting the downloading of this file")
    if config["distribution"]["distributor"] == "multiprocessing" or called_from_slurm:
        distributor_fn = multiprocessing_distributor if stage not in ["whisper", "caption"] else no_distributor
        called_from_slurm = "GLOBAL_RANK" in os.environ
    elif config["distribution"]["distributor"] == "pyspark":
        distributor_fn = pyspark_distributor
    elif config["distribution"]["distributor"] == "slurm":
        worker_args = {key: local_args[key] for key in local_args if not key.startswith("slurm")}
        slurm_args = config["distribution"]["distributor_args"]

        distributor_fn = SlurmDistributor(worker_args=worker_args, **slurm_args)
    elif config["distribution"]["distributor"] == "no_distributor":
        distributor_fn = no_distributor
    else:
        raise ValueError(f"Distributor {config['distribution']['distributor']} not supported")

    distributor_fn(
        config["distribution"]["processes_count"],
        worker,
        shard_iterator,
        config["distribution"]["subjob_size"],
        max_shard_retry,
    )
    logger_process.join()
    if not called_from_slurm:
        fs.rm(run_tmp_dir, recursive=True)


def main():
    args = get_args()
    video2dataset(args)


if __name__ == "__main__":
    main()
