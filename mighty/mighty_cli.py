from mighty.mighty_runners.factory import get_runner_class
import hydra


@hydra.main("./configs", "base", version_base=None)
def run_mighty(cfg):
    # Make runner
    runner_cls = get_runner_class(cfg.runner)
    runner = runner_cls(cfg)
    # Execute run
    train_result, eval_result = runner.run()
    # TODO: pretty print results
    print("hello")
