import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rallyrobopilot.genetic_settings import GeneticSettings

results_dir: Path = Path(__file__).parent.parent / "results"
confs: list[str] = os.listdir(results_dir)
total_df: pd.DataFrame = pd.DataFrame()

configs: dict[str, GeneticSettings] = {}

for i, conf in enumerate(confs):
    conf_dir: Path = results_dir / conf

    seeds: list[str] = os.listdir(conf_dir)
    df: pd.DataFrame = pd.DataFrame()
    for j, seed in enumerate(seeds):
        df_tmp: pd.DataFrame = pd.read_csv(conf_dir / seed / "stats.csv")
        df_tmp["Seed"] = seed.split("_")[1]
        df = pd.concat([df, df_tmp], ignore_index=True)
    df["conf"] = conf.split("_")[1]
    settings: GeneticSettings = GeneticSettings.load(conf_dir / seeds[0] / "settings.json")
    configs[conf] = settings
    df["Configuration"] = f"{settings.mutation_strategy.construction}, {settings.mutation_strategy.selection}"

    total_df = pd.concat([total_df, df], ignore_index=True)

sns.lineplot(total_df, x="gen", y="min_steps", hue="Configuration")
plt.title("Evolution of fastest individual (sampled on 4 different seeds)")
plt.xlabel("Generation")
plt.ylabel("Minimum steps")
plt.show()
