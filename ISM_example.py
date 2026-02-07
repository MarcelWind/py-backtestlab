import matplotlib.pyplot as plt
from stratlab.data.dbnomics import download_dbnomics_catalog

# Fetch all ISM series
mfg = ["ism_pmi", "ism_production", "ism_new_orders", "ism_employment",
       "ism_prices", "ism_inventories", "ism_supplier_deliveries",
       "ism_backlog", "ism_new_export_orders", "ism_imports"]

svc = ["ism_services_pmi", "ism_services_business",
       "ism_services_new_orders", "ism_services_employment",
       "ism_services_prices"]

df_mfg = download_dbnomics_catalog(mfg)
df_svc = download_dbnomics_catalog(svc)

# --- Plot ---
fig, axes = plt.subplots(3, 1, figsize=(14, 14))
fig.suptitle("ISM Survey Data (DBnomics)", fontsize=16, fontweight="bold")

# 1) Manufacturing headline
ax = axes[0]
for col in ["ism_pmi", "ism_production", "ism_new_orders", "ism_employment"]:
    if col in df_mfg.columns:
        ax.plot(df_mfg.index, df_mfg[col], linewidth=1.3, label=col.replace("ism_", ""))
ax.axhline(50, color="black", linewidth=0.8, linestyle="--")
ax.set_ylabel("Index")
ax.set_title("Manufacturing — Headline Components")
ax.legend(loc="lower left", fontsize=8, ncol=2)
ax.grid(alpha=0.3)

# 2) Manufacturing secondary
ax = axes[1]
for col in ["ism_prices", "ism_inventories", "ism_supplier_deliveries",
            "ism_backlog", "ism_new_export_orders", "ism_imports"]:
    if col in df_mfg.columns:
        ax.plot(df_mfg.index, df_mfg[col], linewidth=1.2, label=col.replace("ism_", ""))
ax.axhline(50, color="black", linewidth=0.8, linestyle="--")
ax.set_ylabel("Index")
ax.set_title("Manufacturing — Secondary Components")
ax.legend(loc="lower left", fontsize=8, ncol=3)
ax.grid(alpha=0.3)

# 3) Services
ax = axes[2]
for col in df_svc.columns:
    ax.plot(df_svc.index, df_svc[col], linewidth=1.3, label=col.replace("ism_services_", ""))
ax.axhline(50, color="black", linewidth=0.8, linestyle="--")
ax.set_ylabel("Index")
ax.set_title("Non-Manufacturing (Services)")
ax.legend(loc="lower left", fontsize=8, ncol=3)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
