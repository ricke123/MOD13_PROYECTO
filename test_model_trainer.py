# test_model_trainer.py
import sys
import os
sys.path.append('src')

print("🚀 Probando Model Trainer...")
from model_trainer import ModelTrainer

trainer = ModelTrainer()
result = trainer.full_training_pipeline()

if result:
    print("✅ Model Trainer probado exitosamente!")
else:
    print("❌ Error en Model Trainer")