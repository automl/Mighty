import json


def checkpoint_metadata(agent, file, checkpoint_handler, engine):
    to_save = {}
    try:
        from dacbench import AbstractBenchmark

        bench = AbstractBenchmark()
        bench.config = agent.env.config
        to_save["env_config"] = bench.serialize_config()
    except:
        to_save["env_config"] = "Not using DACBench env, no env config saved"

    to_save["timesteps"] = agent.total_steps
    to_save["episodes"] = agent.logger.module_logger["train_performance"].episode
    to_save["model_dir"] = agent.model_dir
    to_save[
        "checkpoint_path"
    ] = f"{to_save['model_dir']}/{checkpoint_handler.filename_prefix}_checkpoint_{to_save['timesteps']}.pt"
    to_save["checkpoint_evaluated"] = False

    with open(file, "a+") as f:
        json.dump(to_save, f)
        f.write("\n")

    checkpoint_handler(engine, to_save=agent._mapping_save_components)
