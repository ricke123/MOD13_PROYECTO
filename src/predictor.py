# src/predictor.py - VERSIÓN CORREGIDA COMPLETA
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

class DemandPredictor:
    def __init__(self, model_path='data/model/random_forest_optimized_model.pkl'):
        """Inicializa el predictor con el modelo entrenado"""
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.model_type = "unknown"
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo entrenado"""
        print(f"🔍 Cargando modelo desde: {self.model_path}")
        
        try:
            # Verificar si el archivo existe
            if not os.path.exists(self.model_path):
                print(f"❌ ERROR: Archivo no encontrado: {self.model_path}")
                print("💡 Ejecuta: python run_pipeline.py")
                self.model = None
                return
            
            print(f"✅ Archivo encontrado ({os.path.getsize(self.model_path)} bytes)")
            
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Identificar tipo de modelo
            self.model_type = type(self.model).__name__
            print(f"✅ Modelo cargado: {self.model_type}")
            
            # Extraer nombres de features
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
                print(f"📊 Features del modelo: {len(self.feature_names)}")
                print(f"   Primeras 10: {self.feature_names[:10]}")
            else:
                print("⚠️  Modelo no tiene feature_names_in_")
                # Usar las 20 features más importantes del EDA
                self.feature_names = [
                    'demand', 'unique_products', 'unique_orders', 'unique_sellers',
                    'unique_customers', 'total_sales', 'total_freight', 'avg_price',
                    'avg_freight', 'price_min', 'price_max', 'conversion_rate',
                    'price_lag_12', 'demand_growth_12m', 'is_may', 'ema_0.7',
                    'demand_growth_3m', 'year', 'price_ma_6', 'seller_concentration'
                ]
                
        except Exception as e:
            print(f"❌ Error cargando modelo: {type(e).__name__}: {e}")
            self.model = None
            self.feature_names = None
    
    def prepare_features(self, input_data):
        """Prepara features para predicción"""
        # Extraer datos del input
        category = input_data.get('product_category', 'electronics')
        historical_demand = input_data.get('historical_demand', [100, 120, 110])
        promotion = input_data.get('promotion_planned', False)
        seasonality = input_data.get('seasonality_factor', 1.0)
        
        # Calcular valores base
        current_demand = np.mean(historical_demand[-3:]) if len(historical_demand) >= 3 else 100
        
        # Crear diccionario con todas las features posibles
        features_dict = {}
        
        # Features básicas de demanda
        features_dict['demand'] = current_demand
        features_dict['unique_products'] = 25  # Valor típico
        features_dict['unique_orders'] = 50
        features_dict['unique_sellers'] = 15
        features_dict['unique_customers'] = 40
        
        # Features temporales
        now = datetime.now()
        features_dict['month_num'] = now.month
        features_dict['year'] = now.year
        features_dict['quarter'] = (now.month - 1) // 3 + 1
        
        # Features de mes específico
        features_dict['is_may'] = 1 if now.month == 5 else 0
        features_dict['is_november'] = 1 if now.month == 11 else 0
        features_dict['is_october'] = 1 if now.month == 10 else 0
        features_dict['is_august'] = 1 if now.month == 8 else 0
        
        # Features trigonométricas
        features_dict['month_sin'] = np.sin(2 * np.pi * now.month / 12)
        features_dict['month_cos'] = np.cos(2 * np.pi * now.month / 12)
        features_dict['quarter_cos'] = np.cos(2 * np.pi * ((now.month - 1) // 3) / 4)
        
        # Features de crecimiento
        if len(historical_demand) >= 2:
            features_dict['demand_growth_1m'] = (historical_demand[-1] / historical_demand[-2] - 1) if historical_demand[-2] != 0 else 0.05
        else:
            features_dict['demand_growth_1m'] = 0.05
        
        if len(historical_demand) >= 4:
            recent = np.mean(historical_demand[-3:]) if len(historical_demand) >= 3 else current_demand
            older = np.mean(historical_demand[-6:-3]) if len(historical_demand) >= 6 else historical_demand[0] if historical_demand else current_demand
            features_dict['demand_growth_3m'] = (recent / older - 1) if older != 0 else 0.08
        else:
            features_dict['demand_growth_3m'] = 0.08
        
        features_dict['demand_growth_12m'] = 0.12
        
        # Features de precio
        price_base = 100.0 if category == 'electronics' else 80.0
        features_dict['avg_price'] = price_base
        features_dict['price_min'] = price_base * 0.8
        features_dict['price_max'] = price_base * 1.5
        features_dict['price_ma_6'] = price_base * 0.96
        features_dict['price_ma_12'] = price_base * 0.93
        features_dict['price_lag_12'] = price_base * 0.90
        
        # Features estadísticas
        if len(historical_demand) >= 3:
            features_dict['demand_std_3'] = np.std(historical_demand[-3:])
            features_dict['demand_min_3'] = min(historical_demand[-3:])
            features_dict['demand_max_3'] = max(historical_demand[-3:])
        else:
            features_dict['demand_std_3'] = 10
            features_dict['demand_min_3'] = 80
            features_dict['demand_max_3'] = 120
        
        # EMA
        if len(historical_demand) >= 2:
            features_dict['ema_0.7'] = historical_demand[-1] * 0.7 + historical_demand[-2] * 0.3
        else:
            features_dict['ema_0.7'] = current_demand
        
        # Features de negocio
        features_dict['conversion_rate'] = 0.03
        features_dict['seller_concentration'] = 0.25
        features_dict['freight_to_sales_ratio'] = 0.08
        features_dict['total_freight'] = current_demand * 8
        features_dict['total_sales'] = current_demand * price_base
        features_dict['category_demand_mean'] = current_demand * 0.9
        features_dict['category_demand_std'] = current_demand * 0.2
        features_dict['demand_vs_category_avg'] = 1.0
        
        # Features de momentum
        features_dict['demand_momentum_3m'] = 0.05
        features_dict['sales_momentum_3m'] = 0.06
        
        # Ajustar por promoción y estacionalidad
        if promotion:
            features_dict['demand'] *= 1.3
            features_dict['demand_growth_1m'] *= 1.2
            features_dict['demand_growth_3m'] *= 1.2
        
        if seasonality != 1.0:
            features_dict['demand'] *= seasonality
            features_dict['total_sales'] *= seasonality
        
        # Crear DataFrame con todas las features
        features_df = pd.DataFrame([features_dict])
        
        # Si tenemos feature_names específicos del modelo, reordenar
        if self.feature_names is not None:
            # Añadir features faltantes con 0
            for feature in self.feature_names:
                if feature not in features_df.columns:
                    features_df[feature] = 0.0
            
            # Reordenar columnas
            features_df = features_df[self.feature_names]
        
        return features_df
    
    def predict(self, input_data):
        """Realiza predicción y devuelve diccionario con metadatos"""
        print(f"\n🎯 INICIANDO PREDICCIÓN para {input_data.get('product_category', 'unknown')}")
        
        try:
            # Verificar si el modelo está cargado
            if self.model is None:
                print("⚠️  Modelo no cargado, usando simulación")
                simulated_value = self._simulate_prediction(input_data)
                return {
                    "predicted_demand": float(simulated_value),
                    "model_used": "simulation",
                    "confidence": "media",
                    "confidence_score": 0.7,
                    "message": "Modelo no disponible, usando simulación",
                    "success": False
                }
            
            # Preparar features
            features = self.prepare_features(input_data)
            print(f"📊 Features preparadas: {features.shape}")
            
            # Hacer predicción
            raw_prediction = self.model.predict(features)[0]
            predicted_value = max(0, float(raw_prediction))
            print(f"🧮 Predicción obtenida: {predicted_value:.2f}")
            
            # Calcular confianza
            confidence_score = 0.8  # Valor por defecto para Random Forest
            
            # Ajustar confianza basado en features informativas
            if hasattr(features, 'shape'):
                non_zero = (features != 0).sum().sum()
                total = features.size
                if total > 0:
                    info_ratio = non_zero / total
                    confidence_score = max(0.5, min(0.95, info_ratio))
            
            if confidence_score > 0.8:
                confidence = "alta"
            elif confidence_score > 0.6:
                confidence = "media"
            else:
                confidence = "baja"
            
            # Crear resultado como diccionario
            result = {
                "predicted_demand": round(predicted_value, 2),
                "model_used": self.model_type.lower(),
                "confidence": confidence,
                "confidence_score": round(confidence_score, 2),
                "input_features": len(features.columns) if hasattr(features, 'shape') else 0,
                "message": f"Predicción usando {self.model_type}",
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"✅ Predicción completada: {predicted_value:.2f} unidades ({confidence})")
            return result
            
        except Exception as e:
            print(f"❌ Error en predicción: {type(e).__name__}: {e}")
            
            # Fallback a simulación
            simulated_value = self._simulate_prediction(input_data)
            return {
                "predicted_demand": float(simulated_value),
                "model_used": "simulation",
                "confidence": "baja",
                "confidence_score": 0.5,
                "message": f"Error: {str(e)[:100]}... Usando simulación",
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def _simulate_prediction(self, input_data):
        """Predicción simulada (fallback)"""
        base_demand = {
            'electronics': 150,
            'home_appliances': 120, 
            'furniture': 80,
            'computers': 130,
            'housewares': 90
        }
        
        category = input_data.get('product_category', 'electronics')
        historical_demand = input_data.get('historical_demand', [100, 120, 110])
        promotion = input_data.get('promotion_planned', False)
        seasonality = input_data.get('seasonality_factor', 1.0)
        
        base = base_demand.get(category, 100)
        avg_historical = np.mean(historical_demand) if len(historical_demand) > 0 else 100
        
        prediction = (base * 0.3 + avg_historical * 0.7)
        
        if promotion:
            prediction *= 1.3
        
        prediction *= seasonality
        
        # Pequeña variación aleatoria
        prediction *= np.random.uniform(0.95, 1.05)
        
        return max(0, round(prediction, 2))

# ============================================================================
# PRUEBA
# ============================================================================
if __name__ == "__main__":
    print("🧪 PRUEBA DEL PREDICTOR CORREGIDO")
    print("=" * 60)
    
    predictor = DemandPredictor()
    
    test_data = {
        'product_category': 'electronics',
        'historical_demand': [100, 120, 110, 130, 125],
        'promotion_planned': True,
        'seasonality_factor': 1.2
    }
    
    print(f"\n📝 Datos de prueba:")
    for key, value in test_data.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    
    result = predictor.predict(test_data)
    
    print("\n📊 RESULTADO:")
    for key, value in result.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ Prueba completada")