#!/usr/bin/env python3
"""
Script principal para ejecutar el pipeline completo
"""

import sys
import os

# Añadir src al path de forma absoluta
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

def main():
    # Importar después de añadir al path
    from main import main as pipeline_main
    
    print("🚀 Ejecutando Pipeline de Olist Demand Forecasting")
    print("=" * 50)
    
    try:
        result = pipeline_main()
        if result is not None:
            print("\n✅ Pipeline ejecutado exitosamente!")
            print(f"📊 Dataset final: {result.shape}")
        else:
            print("\n❌ Pipeline no pudo completarse")
        return result
        
    except Exception as e:
        print(f"\n❌ Error en el pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()


