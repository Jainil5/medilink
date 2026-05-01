import pandas as pd
import matplotlib.pyplot as plt

# ===== LOAD DATA =====
df = pd.read_csv("/Users/jainil/Documents/development/medilink/app/services/outputs/metrics/final_report.csv")

# Sort for better visualization
df = df.sort_values(by="accuracy", ascending=False)

# =========================================================
# 1. ACCURACY COMPARISON
# =========================================================
plt.figure()
plt.bar(df["model"], df["accuracy"])
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("accuracy_plot.png")
plt.show()


# =========================================================
# 2. PRECISION / RECALL / F1 (MACRO)
# =========================================================
plt.figure()
x = range(len(df))

plt.plot(x, df["precision_macro"], marker='o', label="Precision")
plt.plot(x, df["recall_macro"], marker='o', label="Recall")
plt.plot(x, df["f1_macro"], marker='o', label="F1 Score")

plt.xticks(x, df["model"], rotation=45)
plt.title("Macro Metrics Comparison")
plt.xlabel("Models")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig("macro_metrics.png")
plt.show()


# =========================================================
# 3. WEIGHTED METRICS COMPARISON
# =========================================================
plt.figure()

plt.plot(x, df["precision_weighted"], marker='o', label="Precision Weighted")
plt.plot(x, df["recall_weighted"], marker='o', label="Recall Weighted")
plt.plot(x, df["f1_weighted"], marker='o', label="F1 Weighted")

plt.xticks(x, df["model"], rotation=45)
plt.title("Weighted Metrics Comparison")
plt.xlabel("Models")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig("weighted_metrics.png")
plt.show()


# =========================================================
# 4. MODEL RANKING (ACCURACY)
# =========================================================
plt.figure()

plt.barh(df["model"], df["accuracy"])
plt.title("Model Ranking by Accuracy")
plt.xlabel("Accuracy")
plt.ylabel("Models")
plt.tight_layout()
plt.savefig("ranking_plot.png")
plt.show()


# =========================================================
# 5. ENSEMBLE IMPROVEMENT (VERY IMPORTANT 🔥)
# =========================================================
best_base = df[df["model"] != "ensemble"]["accuracy"].max()
ensemble_acc = df[df["model"] == "ensemble"]["accuracy"].values[0]

plt.figure()

plt.bar(["Best Base Model", "Ensemble"], [best_base, ensemble_acc])

plt.title("Ensemble vs Best Base Model")
plt.ylabel("Accuracy")

# Annotate improvement
improvement = ensemble_acc - best_base
plt.text(1, ensemble_acc, f"+{improvement:.3f}", ha='center')

plt.tight_layout()
plt.savefig("ensemble_improvement.png")
plt.show()