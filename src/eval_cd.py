# Cite https://github.com/mirkobunse/critdd
from src.rl_utils import get_project_folder
import pandas as pd
from collections import defaultdict
from critdd import Diagrams


def load_results(metric):
    results = defaultdict(list)
    models = {"Ours": get_project_folder() / "runs/clf/",
              "VIPER": get_project_folder() / "runs/clf/",
              "PW-Net": get_project_folder() / "runs/pwnet",
              "BB (PPO)": get_project_folder() / "runs/train",
              "SDT": get_project_folder() / "runs/sdt"}

    for key, model in models.items():
        for env in ("CarRacing-v2",
                    "PongNoFrameskip-v4",
                    "MsPacmanNoFrameskip-v4",
                    "BreakoutNoFrameskip-v4",
                    ):
            version = 1 if "VIPER" in key else 0
            if "ppo" in key.lower():
                file_path = model / f"ppo__{env}" / \
                    f"version_{version}/result.csv"
            else:
                file_path = model / env / f"version_{version}/result.csv"
            file = pd.read_csv(file_path)

            results[key] += file[metric].to_list()

    return results


def main():
    results = []
    for metric in ("return", "accuracy", "adjusted_accuracy"):
        result = load_results(metric)

        df = pd.DataFrame(result)

        results.append(df.to_numpy())

    # create a CD diagram from the Pandas DataFrame
    diagram = Diagrams(
        results,
        treatment_names=df.columns,
        diagram_names=("Return", "Acc", "Adj Acc"),
        maximize_outcome=True
    )

    # export the diagram to a file
    diagram.to_file(
        "experiment-data/cd.pdf",
        alpha=.05,
        adjustment="holm",
        reverse_x=True,
    )


if __name__ == "__main__":
    main()
