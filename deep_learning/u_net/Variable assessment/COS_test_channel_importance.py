import pandas as pd
import os

# ------------------------------------------------------------------
# INPUT FILES (your fold outputs)
# ------------------------------------------------------------------

fold1 = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Channel_Occlusion_Sensitivity\fold_results\fold_1\fold1_channel_occlusion_results.csv"
fold2 = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Channel_Occlusion_Sensitivity\fold_results\fold_2\fold2_channel_occlusion_results.csv"
fold3 = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Channel_Occlusion_Sensitivity\fold_results\fold_3\fold3_channel_occlusion_results.csv"

# ------------------------------------------------------------------
# OUTPUT LOCATION
# ------------------------------------------------------------------

out_dir = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Channel_Occlusion_Sensitivity\summary"
out_file = os.path.join(out_dir, "channel_occlusion_importance_summary.csv")

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------

df1 = pd.read_csv(fold1)
df2 = pd.read_csv(fold2)
df3 = pd.read_csv(fold3)

df = pd.concat([df1, df2, df3])

# ------------------------------------------------------------------
# COMPUTE MEAN IMPORTANCE
# ------------------------------------------------------------------

summary = (
    df.groupby("channel_indices")["macro_f1_drop"]
    .mean()
    .reset_index()
)

summary = summary.rename(columns={
    "channel_indices": "Channel",
    "macro_f1_drop": "Mean_macro_F1_drop"
})

# ------------------------------------------------------------------
# CLASSIFICATION RULE
# ------------------------------------------------------------------

def classify(x):
    if x >= 0.015:
        return "Strong"
    elif x >= 0.007:
        return "Moderate"
    else:
        return "Weak"

summary["Importance"] = summary["Mean_macro_F1_drop"].apply(classify)

summary = summary.sort_values(
    "Mean_macro_F1_drop",
    ascending=False
)

# ------------------------------------------------------------------
# SAVE
# ------------------------------------------------------------------

os.makedirs(out_dir, exist_ok=True)

summary.to_csv(out_file, index=False)

print("Saved file:")
print(out_file)
