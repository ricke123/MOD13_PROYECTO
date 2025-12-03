#!/usr/bin/env python3
"""
Diagnóstico del output del predictor
"""
import sys
import os
sys.path.append('src')

from predictor import DemandPredictor

print("🔍 DIAGNÓSTICO DEL OUTPUT DEL PREDICTOR")
print("=" * 60)

# Crear predictor
predictor = DemandPredictor()

# Datos de prueba
test_data = {
    'product_category': 'electronics',
    'historical_demand': [100, 120, 110, 130, 125],
    'promotion_planned': True,
    'seasonality_factor': 1.2
}

print("\n📝 Datos de prueba:")
print(test_data)

print("\n🧠 Ejecutando predicción...")
try:
    result = predictor.predict(test_data)
    
    print("\n📊 RESULTADO:")
    print(f"Tipo: {type(result)}")
    print(f"Valor: {result}")
    print(f"Longitud: {len(str(result)) if hasattr(result, '__len__') else 'N/A'}")
    
    if isinstance(result, dict):
        print("\n🔑 Claves del diccionario:")
        for key, value in result.items():
            print(f"  {key}: {value} ({type(value).__name__})")
    elif isinstance(result, str):
        print("\n📄 Contenido del string:")
        print(result[:500])  # Primeros 500 caracteres
    else:
        print(f"\n⚠️ Tipo inesperado: {type(result)}")
        
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()