# test_imports.py
import sys
import os

# Añadir src al path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

print("🧪 Probando imports...")

try:
    from config import Config
    print("✅ config.py - OK")
    
    from data_loader import DataLoader
    print("✅ data_loader.py - OK")
    
    from feature_engineer import FeatureEngineer
    print("✅ feature_engineer.py - OK")
    
    from preprocessor import DataPreprocessor
    print("✅ preprocessor.py - OK")
    
    from utils import check_data_files
    print("✅ utils.py - OK")
    
    from main import main
    print("✅ main.py - OK")
    
    print("\n🎉 Todos los imports funcionan correctamente!")
    
except ImportError as e:
    print(f"❌ Error de import: {e}")