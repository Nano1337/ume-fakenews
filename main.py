import argparse

parser = argparse.ArgumentParser(description="which directory to run")
parser.add_argument("--dir", type=str, default=None, help="directory to run")
arg = parser.parse_args()


if arg.dir == "fakeddit":
    from fakeddit.run_trainer import run_training
else: 
    raise NotImplementedError("Please specify a directory to run")

run_training()