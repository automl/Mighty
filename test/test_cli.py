import os
import shutil


class TestMightCLI:
    def test_entrypoint(self):
        exit_status = os.system("mighty --help")  # noqa:F841
        # assert exit_status == 0

    def test_run(self):
        exit_status = os.system("mighty num_steps=100 output_dir=test_cli")  # noqa:F841
        # assert exit_status == 0
        # shutil.rmtree('test_cli')

    def test_run_from_file(self):
        exit_status = os.system(
            "python mighty/run_mighty.py num_steps=100 output_dir=test_cli"
        )
        assert exit_status == 0
        shutil.rmtree("test_cli")
