import pandas as pd
import numpy as np

df = pd.read_csv("genshin_relevance_mean.csv")

# 1) Normalize roles (extra safety)
role_map = {
    "DPS": "Main DPS",
    "Main DPS": "Main DPS",
    "Sub DPS": "Sub-DPS",
    "Sub-DPS": "Sub-DPS",
    "Support": "Support",
}
df["role"] = df["role"].map(lambda x: role_map.get(str(x).strip(), str(x).strip()))

# 2) Normalize character names (aliases)
name_alias = {
    "Childe": "Tartaglia",
    "Kuki Shinobu": "Shinobu",          # or reverse; pick ONE canonical
    "Yumemizuki Mizuki": "Mizuki",      # or reverse
}
df["name"] = df["name"].replace(name_alias)

# 3) Optional: penalize missing source (uncertainty)
# If one source missing -> multiply relevance by 0.95 (example)
missing_one = df["game8_score"].isna() ^ df["genshin.gg_score"].isna()
df["relevance_mean_adj"] = df["relevance_mean"]
df.loc[missing_one, "relevance_mean_adj"] = df.loc[missing_one, "relevance_mean"] * 0.95

# 4A) Final table per (name, role): recompute by grouping in case merges created duplicates after aliasing
agg_role = (
    df.groupby(["name", "role"], as_index=False)
      .agg({
          "game8_score": "max",
          "genshin.gg_score": "max",
          "relevance_mean": "max",
          "relevance_mean_adj": "max",
          "game8_tier": "first",
          "genshin.gg_tier": "first",
      })
      .sort_values(["relevance_mean_adj", "relevance_mean", "name"], ascending=[False, False, True])
)

agg_role.to_csv("genshin_relevance_clean_role.csv", index=False, encoding="utf-8")

# 4B) Final table per character (ignore role) - choose best role relevance
agg_char_best = (
    agg_role.groupby("name", as_index=False)
            .agg({
                "relevance_mean_adj": "max",
                "relevance_mean": "max",
                "game8_score": "max",
                "genshin.gg_score": "max",
            })
            .sort_values(["relevance_mean_adj", "name"], ascending=[False, True])
)

agg_char_best.to_csv("genshin_relevance_clean_character_best.csv", index=False, encoding="utf-8")

print("Saved:\n- genshin_relevance_clean_role.csv\n- genshin_relevance_clean_character_best.csv")
