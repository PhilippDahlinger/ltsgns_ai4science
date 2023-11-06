from lts_gns.recording.loggers.logger_util.wandb_util import get_job_type
from lts_gns.util.config_dict import ConfigDict


def _insert_recording_structure(cw_config: ConfigDict, repetition: int) -> bool:
    rep_log_path = cw_config.get("_rep_log_path")
    experiment_name = cw_config.get("_experiment_name")
    job_name = cw_config.get("name")


    cw_config["params"]["_recording_structure"] = {
        "_groupname": experiment_name,   # wandb group name
        "_runname": experiment_name + "_" + str(repetition),  # wandb run name
        "_recording_dir": rep_log_path,   # path to the cw2 logging folder
        "_job_name": job_name
    }
    return True

