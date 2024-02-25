"""Runs a simple sythetic data simulation and dumps to a table.

python docker_main.py --alpha=0.05 --beta=0.2 --seed=42 --cut_type=BL --p0=0.5 --p1=0.5 --n_periods=100 \
    --m_null=100 --m_alt=100 --hyp_type=pois --sim_reps=1000 --seed=42 --host=localhost --database=MultSeq \
     --username=root --password=1234 --run_id=1 
"""

import argparse
import os
import pandas as pd
import numpy as np
import sqlalchemy
from utils import simulation_orchestration
import datetime
import uuid
import copy

# Imports the Cloud Logging client library
import google.cloud.logging

# Instantiates a client
client = google.cloud.logging.Client()

# Retrieves a Cloud Logging handler based on the environment
# you're running in and integrates the handler with the
# Python logging module. By default this captures all logs
# at INFO level and higher
client.setup_logging()
import logging


def connect_with_connector() -> sqlalchemy.engine.base.Engine:
    """
    Initializes a connection pool for a Cloud SQL instance of MySQL.

    Uses the Cloud SQL Python Connector package.
    """
    from google.cloud.sql.connector import Connector, IPTypes
    import pymysql

    # Note: Saving credentials in environment variables is convenient, but not
    # secure - consider a more secure solution such as
    # Cloud Secret Manager (https://cloud.google.com/secret-manager) to help
    # keep secrets safe.

    instance_connection_name = os.environ[
        "INSTANCE_CONNECTION_NAME"
    ]  # e.g. 'project:region:instance'
    db_user = os.environ["DB_USER"]  # e.g. 'my-db-user'
    db_pass = os.environ["DB_PASS"]  # e.g. 'my-db-password'
    db_name = os.environ["DB_NAME"]  # e.g. 'my-database'

    ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC

    connector = Connector(ip_type)

    def getconn() -> pymysql.connections.Connection:
        conn: pymysql.connections.Connection = connector.connect(
            instance_connection_name,
            driver="pymysql",
            user=db_user,
            password=db_pass,
            db=db_name,
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn,
        # ...
    )
    return pool


def get_connection(config):
    cloud = config.pop("cloud", False)
    if not cloud:
        # Get SQL connection
        username = config.pop("username", None)
        password = config.pop("password", None)
        host = config.pop("host", None)
        port = config.pop("port", None)
        database = config.pop("database", None)
        database_url = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        engine = sqlalchemy.create_engine(database_url)

        seed = config.pop("seed", None)

        run_id = config.pop("run_id", 0)
        total_runs = 1
        cloud_execution = ""
    else:
        engine = connect_with_connector()

        run_id = os.environ.get("CLOUD_RUN_TASK_INDEX")
        total_runs = os.environ.get("CLOUD_RUN_TASK_COUNT")
        cloud_execution = os.environ.get("CLOUD_RUN_EXECUTION")
        assert run_id is not None, "Run ID must be provided"
        assert total_runs is not None, "Total runs must be provided"
        seed = run_id
    return {
        "engine": engine,
        "run_id": run_id,
        "total_runs": total_runs,
        "cloud_execution": cloud_execution,
        "seed": seed,
    }


def main(parameter_config, run_config):

    # Get initiation time
    now = datetime.datetime.now()
    run_start = pd.to_datetime(now)
    run_str = uuid.uuid4().hex
    # Extract pamarameters
    param_dict = dict(
        alpha=parameter_config.pop("alpha", 0.05),
        beta=parameter_config.pop("beta", 0.2),
        cut_type=parameter_config.pop("cut_type", "BL"),
        theta0=parameter_config.pop("theta0", 0.5),
        theta1=parameter_config.pop("theta1", 0.5),
        n_periods=parameter_config.pop("n_periods", 100),
        m_null=parameter_config.pop("m_null", 100),
        m_alt=parameter_config.pop("m_alt", 100),
        hyp_type=parameter_config.pop("hyp_type", "pois"),
        rho=parameter_config.pop("rho", -0.5),
        # max_magnitude=parameter_config.pop("max_magnitude", 10.0),
        extra_params=parameter_config.pop("extra_params", {}),
    )
    if param_dict["extra_params"] is None:
        param_dict["extra_params"] = {}
    hyp_type = param_dict["hyp_type"]
    if hyp_type == "pois":
        generating_params = {}
    elif hyp_type == "binom":
        generating_params = {"p0": 0.5, "p1": 0.5}
    else:
        raise ValueError(f"Unknown hypothesis type {hyp_type}")
    ## --args=--alpha=0.05,--beta=0.2,--cut_type=BL,--p0=0.5,--p1=0.5,--m_null=3,--m_alt=7,--hyp_type=pois,--sim_reps=1000,--cloud

    sim_reps = parameter_config.pop("sim_reps", None)
    if sim_reps is None:
        sim_reps = 1000

    # Add more configuration options as needed

    engine = run_config.pop("engine", None)
    run_id = run_config.pop("run_id", None)
    cloud_execution = run_config.pop("cloud_execution", "")
    seed = run_config.pop("seed", None)
    total_runs = run_config.pop("total_runs", None)

    if seed is not None:
        try:
            seed = int(seed)
            np.random.seed(seed)
        except ValueError as ex:
            logging.warn(f"Seed {seed} is not an integer. Using default seed. Error message: {ex}")

    assert run_id is not None, "Run ID must be provided"
    try:
        sim_reps = int(sim_reps)
        total_runs = int(total_runs)
        run_id = int(run_id)
    except ValueError as ex:
        print(f"Error converting sim_reps or total_runs to int. Error message: {ex}")
        raise

    if cloud_execution:
    
        n_reps_per_job = sim_reps // total_runs
        n_jobs_with_extras = sim_reps - (total_runs * n_reps_per_job)
        if run_id < n_jobs_with_extras:
            job_reps = n_reps_per_job + 1
            start_rep = run_id * job_reps
        else:
            job_reps = n_reps_per_job
            start_rep = run_id * job_reps + n_jobs_with_extras
    
    else:
        start_rep = 0
        job_reps = sim_reps

    if run_id == 0:
        metadata_df = pd.DataFrame(
            [
                pd.Series(
                    {
                        "cloud_execution": cloud_execution,
                        "run_start": run_start,
                        "total_runs": total_runs,
                        "sim_reps": sim_reps,
                    }
                )
            ]
        ).set_index("cloud_execution")
        stringified_params = {}
        for kk, vv in param_dict.items():
            if kk=="extra_params":
                stringified_params[kk] = str(vv)
            else:
                stringified_params[kk] = vv
        param_df = pd.DataFrame(
            [pd.Series(stringified_params)],
            index=pd.Index([cloud_execution], name="cloud_execution"),
        )
        md_df_str = f"Metadata: {metadata_df}"
        md_dt_str = f"Metadata dtypes: {metadata_df.dtypes}"
        param_df_str = f"Parameters: {param_df}"
        param_df_dt_str = f"Parameters dtypes: {param_df.dtypes}"
        logging.info("Writing metadata and parameters to database")
        logging.info(md_df_str)
        logging.info(md_dt_str)
        logging.info(param_df_str)
        logging.info(param_df_dt_str)
        print(md_df_str)
        print(md_dt_str)
        print(param_df_str)
        print(param_df_dt_str)
        param_df.to_sql("simulation_params", con=engine, if_exists="append", index=True)
        metadata_df.to_sql("simulation_metadata", con=engine, if_exists="append", index=True)

    # Call your simulation function

    # Extract metadata

    df = simulation_orchestration.mc_sim_and_analyze_synth_data(
        **param_dict,
        sim_reps=job_reps,
        # Everything after this is hardwired right now.
        # record_interval=100,
        m0_known=False,
        scale_fdr=True,
        interleaved=False,
        undershoot_prob=0.2,
        fin_par=False,
        fh_sleep_time=60,
        do_iterative_cutoff_MC_calc=False,
        stepup=False,
        fh_cutoff_normal_approx=False,
        fh_cutoff_imp_sample=True,
        fh_cutoff_imp_sample_prop=0.5,
        fh_cutoff_imp_sample_hedge=0.9,
        load_data=None,
        divide_cores=None,
        split_corr=False,
        rho1=None,
        rand_order=False,
        cummax=False,
        analysis_func=simulation_orchestration.compute_fdp,
    )
    df["seed"] = seed
    df["run_id"] = run_id
    df["run_start"] = run_start
    df["mc_iter"] = np.arange(start_rep, start_rep + job_reps)
    df["cloud_execution"] = cloud_execution

    # Convert DataFrame to SQL
    df.to_sql("simulation_results", con=engine, if_exists="append", index=False)

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run a simulation and store the results in a database."
    )
    # Add arguments for each of the configuration options your simulation requires
    parser.add_argument("--alpha", type=float, help="FDR level")
    parser.add_argument("--beta", type=float, help="Optional FNR level")
    parser.add_argument("--cut_type", type=str, help="Type of cut")
    parser.add_argument("--theta0", type=float, help="Null hypothesis parameter value.")
    parser.add_argument("--theta1", type=float, help="Alternative hypothesis parameter value.")
    parser.add_argument("--extra_params", nargs='*', action=ParseKwargs, help="other generating parameters")
    parser.add_argument("--n_periods", type=int, help="Number of periods")
    parser.add_argument("--m_null", type=int, help="Number of null hypotheses")
    parser.add_argument("--m_alt", type=int, help="Number of alternative hypotheses")
    parser.add_argument("--hyp_type", type=str, help="Type of hypothesis")
    parser.add_argument("--sim_reps", type=int, help="Number of simulation repetitions")
    # Metadata/launch specific
    parser.add_argument("--cloud", action="store_true", help="Run in cloud mode")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--host", type=str, help="Database host", default=3306)
    parser.add_argument("--port", type=int, help="Database port")
    parser.add_argument("--database", type=str, help="Database name")
    parser.add_argument("--username", type=str, help="Database username")
    parser.add_argument("--password", type=str, help="Database password")
    parser.add_argument("--run_id", type=int, help="Unique run identifier")

    # Add more arguments as needed

    args = parser.parse_args()

    # Convert parsed arguments to a dictionary
    config = vars(args)
    for k, v in config.items():
        print(k, v, type(v))
    connection_config = get_connection(config)
    param_config = config
    main(param_config, connection_config)
