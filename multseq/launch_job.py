# --configfile=../Configs/test.yaml
import yaml
from dotenv import load_dotenv
import argparse
import subprocess
import pathlib
import os
from google.cloud import run_v2
import asyncio

load_dotenv("../.env")

parser = argparse.ArgumentParser(description="Launch a job")
parser.add_argument("--configfile", type=str, help="Path to the config file")
parser.add_argument(
    "--configname",
    type=str,
    default=None,
    help="Name of the configuration. If blank, will launch a job for each model",
)

parser.add_argument(
    "--project",
    type=str,
    help="Google Cloud Project ID",
    default=os.getenv("PROJECT_ID"),
)
parser.add_argument(
    "--region", type=str, help="Google Cloud Region", default=os.getenv("REGION")
)
parser.add_argument(
    "--jobname", type=str, help="Name of the job", default="main-launcher"
)


def build_args(config):
    args = ["--cloud"]
    for k, v in config.items():
        args.append(f"--{k}={v}")
    return args


ACCEPTABLE_CONFIG_KEYS = [
    "alpha",
    "beta",
    "cut_type",
    "theta0",
    "theta1",
    "m_null",
    "m_alt",
    "hyp_type",
    "sim_reps",
    "reps_per_task",
    "error_control",
    "n_periods",
    "rho",
    "extra_params",
]
    
def validate_config(config):
    """
    Validates the configuration dictionary for a job.

    Args:
        config: dictionary of key-value pairs to pass as arguments to the job
    """
    unknown_keys = set(config.keys()).difference(ACCEPTABLE_CONFIG_KEYS)
    if  unknown_keys:
        raise ValueError(f"Invalid keys in config: {unknown_keys}")
    
# async def launch_job(project_id, region, job_name, config, run_name=None):
def launch_job(project_id, region, job_name, config, run_name=None):
    """
    Launches a Cloud Run Job with specific arguments.

    Args:
        project_id: Your Google Cloud project ID.
        region: The region where the job runs.
        job_name: The name of the job to launch.
        config: dictionary of key-value pairs to pass as arguments to the job
    """
    if "reps_per_task" in config:
        num_tasks = int(config["sim_reps"] / config.pop("reps_per_task"))
    else:
        num_tasks = 10
    # Create a RunServiceClient
    # client = run_v2.JobsAsyncClient()

    client = run_v2.JobsClient()
    # Build the set of commandline args for the job
    if run_name is not None:
        config["run_name"] = run_name
    args = build_args(config)
    # print(" ".join(args))
    # raise ValueError()
    # Create a set of overrides for the job to inject the commandline args and the number of tasks
    ov = run_v2.RunJobRequest.Overrides(
        task_count=num_tasks,
        container_overrides=[
            run_v2.RunJobRequest.Overrides.ContainerOverride(
                args=args,
            ),
        ],
    )
    # Create a RunJobRequest
    request = run_v2.RunJobRequest(
        name=f"projects/{project_id}/locations/{region}/jobs/{job_name}",
        overrides=ov,
    )

    # Asynchronously execute the job
    operation = client.run_job(request)

    # Wait for the job to complete and print the status
    # result = (await operation).result()
    result = operation.result()
    print(f"Job {run_name} status: {result}")


# async def main(args):
def main(args):
    configfile = pathlib.Path(args.configfile)
    assert configfile.exists(), f"Config file {configfile} does not exist"
    configname = args.configname

    with open(configfile, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if configname is not None:
        assert (
            configname in config
        ), f"Config name {configname} not found in config file"
        config = {configname: config[configname]}
        print("Running only for config: ", configname)
    else:
        print("Running for all configs")

    job_list = []
    # First validate
    for c_name, c_dict in config.items():
        validate_config(c_dict)
    # Then launch
    for c_name, c_dict in config.items():
        # TODO: do this asynchronously
        job_list.append(launch_job(args.project, args.region, args.jobname, c_dict, run_name=c_name))
    # njobs = len(job_list)
    # print(f"Launching {njobs} jobs")
    # await asyncio.gather(*job_list)

if __name__ == "__main__":
    args = parser.parse_args()
    # asyncio.run(main(args))
    main(args)

