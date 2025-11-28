#!/usr/bin/env python3
"""
Ejecución rápida del pipeline completo
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main import run_pipeline

if __name__ == "__main__":
    print("🚀 Ejecutando pipeline completo...")
    run_pipeline()
    print("✅ Pipeline completado exitosamente!")