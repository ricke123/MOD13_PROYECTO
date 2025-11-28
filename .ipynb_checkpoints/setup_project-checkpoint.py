# setup_project.py
import os

def create_project_structure():
    """Crea la estructura completa del proyecto"""
    
    base_dir = "olist-demand-forecasting"
    directories = [
        base_dir,
        os.path.join(base_dir, "src"),
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "data", "raw"),
        os.path.join(base_dir, "data", "processed"),
        os.path.join(base_dir, "notebooks"),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Carpeta creada: {directory}")
    
    print(f"\n🎯 Estructura creada en: {os.path.abspath(base_dir)}")

if __name__ == "__main__":
    create_project_structure()