from src.rl_utils import get_project_folder
from statistics import mean, stdev
import pandas as pd


def main(env: str = "MsPacmanNoFrameskip-v4", version: int = 39):
    file_path = get_project_folder() / "runs/clf" / env / \
        f"version_{version}/result.csv"

    df = pd.read_csv(file_path)

    print("Version", version)
    print(df.mean(), df["return"].std())


if __name__ == "__main__":
    main()
