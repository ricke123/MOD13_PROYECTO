# test_data_updater.py
import sys
import os
sys.path.append('src')

print("🔄 Probando Data Updater...")
from data_updater import DataUpdater

updater = DataUpdater()
new_data = updater.simulate_new_monthly_data(months_to_add=1)

if new_data is not None:
    print(f"✅ Nuevos datos simulados: {new_data.shape}")
    updated_data = updater.update_dataset(new_data)
    updater.save_updated_data(updated_data)
    print("✅ Data Updater probado exitosamente!")
else:
    print("❌ Error en Data Updater")