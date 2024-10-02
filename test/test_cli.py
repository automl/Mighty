import os
import shutil


class TestMightCLI:
    def test_run_from_file(self):
        exit_status = os.system(
            "python mighty/run_mighty.py num_steps=100 output_dir=test_cli"
        )
        assert exit_status == 0
        shutil.rmtree("test_cli")