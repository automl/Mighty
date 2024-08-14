from __future__ import annotations
from typing import Tuple, Dict
from mighty.mighty_runners.mighty_runner import MightyRunner


class MightyOnlineRunner(MightyRunner):
    def run(self) -> Tuple[Dict, Dict]:
        train_results = self.train(self.num_steps)
        eval_results = self.evaluate()
        self.close()
        return train_results, eval_results
