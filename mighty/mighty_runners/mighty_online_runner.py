from __future__ import annotations

from mighty.mighty_runners.mighty_runner import MightyRunner


class MightyOnlineRunner(MightyRunner):
    def run(self):
        train_results = self.train(self.num_steps)
        eval_results = self.evaluate()
        self.close()
        return train_results, eval_results
