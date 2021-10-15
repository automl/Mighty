import json

def checkpoint_metadata(agent, file, checkpoint_handler, engine):
    to_save = {}
    #to_save["env_config"] = agent.env.config
    to_save["timesteps"] = agent.total_steps

    to_save["model_dir"] = agent.model_dir
    to_save["checkpoint_path"] = f"{to_save['model_dir']}/model_{to_save['timesteps']/11}.pt"

    with open(file, "a+") as f:
        json.dump(to_save, f, indent=2)

    checkpoint_handler(engine, to_save=agent._mapping_save_components)
