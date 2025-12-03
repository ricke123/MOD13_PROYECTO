



# api.py - VERSIÓN CORREGIDA PARA WINDOWS
"""
API REST para el sistema de predicción de demanda Olist.
Versión 2.0.1 - Corregida para Windows y Pydantic 2.x
"""
import os
import sys
from datetime import datetime
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import uvicorn
import traceback

# Asegurar que `src` esté en el path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Importar módulos del proyecto
try:
    from src.predictor import DemandPredictor
    from src.business_metrics import BusinessMetrics
    PREDICTOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Error importando módulos: {e}")
    print("💡 Asegúrate de que src/predictor.py y src/business_metrics.py existan")
    PREDICTOR_AVAILABLE = False

# Intentar importar MLflow (puede fallar silenciosamente)
MLFLOW_AVAILABLE = False
mlflow_tracker = None
try:
    from monitoring.mlflow_tracking import MLflowTracker
    MLFLOW_AVAILABLE = True
except ImportError:
    print("ℹ️ MLflow no disponible - continuando sin logging")

# === CONFIGURACIÓN INICIAL ===
app = FastAPI(
    title="Olist Demand Forecasting API",
    description="API para predicción de demanda mensual por categoría",
    version="2.0.1",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Cargar predictor
predictor = None
if PREDICTOR_AVAILABLE:
    try:
        # Intentar varias rutas posibles para el modelo
        possible_paths = [
            'data/model/random_forest_optimized_model.pkl',
            'data/model/xgboost_optimized_model.pkl',
            'model/random_forest_model.pkl',
            'model/demand_model.pkl'
        ]
        
        model_loaded = False
        for model_path in possible_paths:
            if os.path.exists(model_path):
                predictor = DemandPredictor(model_path=model_path)
                if predictor.model:
                    print(f"✅ Modelo cargado desde: {model_path}")
                    print(f"✅ Tipo de modelo: {type(predictor.model).__name__}")
                    model_loaded = True
                    break
        
        if not model_loaded:
            print("⚠️ No se encontró ningún modelo, usando simulación")
            predictor = DemandPredictor()
            predictor.model = None  # Forzar modo simulación
    except Exception as e:
        print(f"❌ Error inicializando predictor: {e}")
        predictor = DemandPredictor()
        predictor.model = None

# Inicializar MLflow si está disponible - CON MANEJO ESPECÍFICO PARA WINDOWS
if MLFLOW_AVAILABLE:
    try:
        # Configurar MLflow manualmente primero para evitar error de Windows
        import mlflow
        
        # Usar path absoluto para Windows
        mlruns_path = os.path.join(os.getcwd(), "mlruns")
        mlflow.set_tracking_uri(f"file:///{mlruns_path}")
        
        # Crear directorio si no existe
        os.makedirs(mlruns_path, exist_ok=True)
        
        # Configurar experimento
        experiment_name = "olist-demand-forecasting"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name, artifact_location=mlruns_path)
        except:
            pass
        
        mlflow.set_experiment(experiment_name)
        
        # Ahora inicializar nuestro tracker
        mlflow_tracker = MLflowTracker()
        print("✅ MLflow tracker inicializado correctamente")
        
    except Exception as e:
        print(f"⚠️ Error inicializando MLflow: {e}")
        print("💡 Continuando sin MLflow...")
        mlflow_tracker = None
else:
    print("ℹ️ MLflow no disponible, continuando sin logging")

# === MODELOS DE DATOS (Pydantic) ===
class PredictionRequest(BaseModel):
    product_category: str = Field(
        default="electronics",
        description="Categoría del producto",
        example="electronics"
        # NOTA: pattern en lugar de regex para Pydantic 2.x
    )
    historical_demand: List[float] = Field(
        default=[100.0, 120.0, 110.0, 130.0, 125.0],
        description="Lista de demandas históricas (últimos meses)",
        min_items=1,
        max_items=24,
        example=[100, 120, 110, 130, 125]
    )
    promotion_planned: bool = Field(
        default=False,
        description="¿Hay promoción planificada el próximo mes?",
        example=True
    )
    seasonality_factor: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Factor de ajuste estacional (0.1–3.0)",
        example=1.2
    )

class PredictionResponse(BaseModel):
    predicted_demand: float
    product_category: str
    confidence: str
    timestamp: str
    status: str
    model_used: str
    confidence_score: Optional[float] = None
    message: Optional[str] = None
    request_id: Optional[str] = None

class BusinessMetricsRequest(BaseModel):
    actual_demand: List[float]
    predicted_demand: List[float]
    categories: Optional[List[str]] = None

class BusinessMetricsResponse(BaseModel):
    accuracy_percentage: float
    roi_percentage: float
    monthly_savings_estimate: float
    overstock_units: float
    stockout_units: float
    inventory_costs_usd: float
    stockout_costs_usd: float

# === FUNCIONES AUXILIARES ===
def generate_request_id() -> str:
    """Genera un ID único para la solicitud"""
    import random
    return f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"

def safe_serialize(obj: Any) -> Any:
    """Convierte valores numpy/bool a tipos nativos JSON-safe"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    else:
        try:
            return str(obj)
        except:
            return "unserializable"

def extract_prediction_value(prediction_result: Union[Dict, float, int, str]) -> float:
    """Extrae el valor numérico de la predicción"""
    try:
        if isinstance(prediction_result, dict):
            # Intentar extraer de varias claves posibles
            for key in ['predicted_demand', 'prediction', 'demand', 'value', 'result']:
                if key in prediction_result:
                    val = prediction_result[key]
                    if isinstance(val, (int, float)):
                        return float(val)
                    else:
                        try:
                            return float(val)
                        except:
                            continue
            
            # Si no encontró en claves específicas, buscar cualquier número
            for val in prediction_result.values():
                if isinstance(val, (int, float)):
                    return float(val)
            
            # Último recurso
            return 100.0
        
        elif isinstance(prediction_result, (int, float)):
            return float(prediction_result)
        
        else:
            # Intentar convertir
            return float(prediction_result)
            
    except Exception as e:
        print(f"⚠️ Error extrayendo valor de predicción: {e}")
        return 100.0  # Valor por defecto seguro

def calculate_data_confidence(historical_demand: List[float]) -> tuple:
    """Calcula confianza basada en calidad de datos históricos"""
    if len(historical_demand) < 2:
        return "baja", 0.3
    
    hist_array = np.array(historical_demand, dtype=float)
    mean_val = np.mean(hist_array)
    std_val = np.std(hist_array)
    
    if mean_val == 0:
        return "baja", 0.3
    
    cv = std_val / mean_val  # Coeficiente de variación
    
    if cv < 0.15 and len(historical_demand) >= 6:
        return "alta", 0.85
    elif cv < 0.3 and len(historical_demand) >= 4:
        return "media", 0.65
    else:
        return "baja", 0.45

def safe_mlflow_logging(request_id: str, input_data: Dict, 
                       prediction_result: Dict, predicted_value: float):
    """Logging seguro en MLflow con manejo de errores"""
    if not mlflow_tracker:
        return
    
    try:
        # Preparar datos para logging
        safe_input = {k: safe_serialize(v) for k, v in input_data.items()}
        
        # Extraer información del resultado
        model_used = prediction_result.get('model_used', 'unknown')
        confidence = prediction_result.get('confidence', 'unknown')
        confidence_score = prediction_result.get('confidence_score', 0.5)
        
        # Intentar logging con el tracker
        with mlflow_tracker.start_run(run_name=f"api_pred_{request_id}"):
            # Log parámetros
            mlflow_tracker.log_params(safe_input)
            
            # Log métricas
            mlflow_tracker.log_metric("predicted_demand", float(predicted_value))
            mlflow_tracker.log_metric("confidence_score", float(confidence_score))
            
            # Log tags
            mlflow_tracker.set_tag("model_used", model_used)
            mlflow_tracker.set_tag("confidence", confidence)
            mlflow_tracker.set_tag("request_id", request_id)
            mlflow_tracker.set_tag("source", "api")
            mlflow_tracker.set_tag("environment", "production")
            mlflow_tracker.set_tag("status", "success")
        
        print(f"📊 MLflow: Predicción logged (ID: {request_id})")
        
    except Exception as e:
        print(f"⚠️ MLflow logging falló (no crítico): {e}")
        # No hacer nada más - el logging no debe romper la API

# === ENDPOINTS ===
@app.get("/")
async def root():
    """Endpoint raíz con información del sistema"""
    if not predictor:
        return {
            "message": "API Olist Forecasting - Predictor no disponible",
            "version": "2.0.1",
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }
    
    model_loaded = predictor.model is not None
    model_type = type(predictor.model).__name__ if model_loaded else "simulation"
    
    return {
        "message": "🚀 Bienvenido a la API de Forecasting Olist",
        "version": "2.0.1",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health", 
            "metrics": "POST /metrics",
            "docs": "GET /docs",
            "status": "GET /status"
        },
        "system_info": {
            "model_loaded": model_loaded,
            "model_type": model_type,
            "mlflow_available": MLFLOW_AVAILABLE and mlflow_tracker is not None,
            "features_count": len(predictor.feature_names) if hasattr(predictor, 'feature_names') and predictor.feature_names else 0
        }
    }

@app.get("/health")
async def health_check():
    """Health check del sistema"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Verificar predictor
    if predictor and predictor.model:
        health_status["components"]["predictor"] = {
            "status": "healthy",
            "model_type": type(predictor.model).__name__,
            "model_path": predictor.model_path
        }
    else:
        health_status["components"]["predictor"] = {
            "status": "degraded",
            "message": "Usando modo simulación"
        }
        health_status["status"] = "degraded"
    
    # Verificar MLflow
    if mlflow_tracker:
        health_status["components"]["mlflow"] = {
            "status": "healthy"
        }
    else:
        health_status["components"]["mlflow"] = {
            "status": "disabled",
            "message": "MLflow no disponible"
        }
    
    return health_status

@app.post("/predict", response_model=PredictionResponse)
async def predict_demand(request: PredictionRequest):
    """
    Predice la demanda futura basada en datos históricos y configuraciones.
    """
    request_id = generate_request_id()
    print(f"\n🎯 Nueva predicción (ID: {request_id})")
    print(f"📝 Categoría: {request.product_category}")
    print(f"📊 Datos históricos: {len(request.historical_demand)} puntos")
    
    try:
        # Verificar que el predictor esté disponible
        if not predictor:
            raise HTTPException(
                status_code=503, 
                detail="Predictor no disponible. Intente nuevamente más tarde."
            )
        
        # Validar categoría (ahora manualmente)
        valid_categories = ["electronics", "home_appliances", "furniture", "computers", "housewares"]
        if request.product_category not in valid_categories:
            raise HTTPException(
                status_code=400,
                detail=f"Categoría inválida. Debe ser una de: {', '.join(valid_categories)}"
            )
        
        # Preparar datos de entrada
        input_data = {
            "product_category": request.product_category,
            "historical_demand": request.historical_demand,
            "promotion_planned": request.promotion_planned,
            "seasonality_factor": request.seasonality_factor
        }
        
        # Obtener predicción
        prediction_result = predictor.predict(input_data)
        
        if not isinstance(prediction_result, dict):
            print(f"⚠️ Formato inesperado de predicción: {type(prediction_result)}")
            prediction_result = {"predicted_demand": prediction_result}
        
        # Extraer valores
        predicted_value = extract_prediction_value(prediction_result)
        
        # Obtener metadatos del resultado
        model_used = prediction_result.get('model_used', 
                    'RandomForest' if predictor.model and hasattr(predictor.model, 'n_estimators')
                    else 'XGBoost' if predictor.model and hasattr(predictor.model, 'get_booster')
                    else 'simulation')
        
        confidence = prediction_result.get('confidence', 'media')
        confidence_score = prediction_result.get('confidence_score', 0.5)
        message = prediction_result.get('message', '')
        
        # Calcular confianza adicional basada en datos
        data_confidence, data_score = calculate_data_confidence(request.historical_demand)
        
        # Combinar confianzas
        final_confidence_score = max(confidence_score, data_score)
        if final_confidence_score >= 0.75:
            final_confidence = "alta"
        elif final_confidence_score >= 0.5:
            final_confidence = "media"
        else:
            final_confidence = "baja"
        
        # Loggear en MLflow (no bloqueante)
        safe_mlflow_logging(request_id, input_data, prediction_result, predicted_value)
        
        # Preparar respuesta
        response = PredictionResponse(
            predicted_demand=round(float(predicted_value), 2),
            product_category=request.product_category,
            confidence=final_confidence,
            timestamp=datetime.now().isoformat(),
            status="success",
            model_used=model_used,
            confidence_score=round(final_confidence_score, 3),
            message=f"{message[:100]}" if message else f"Predicción para {request.product_category}",
            request_id=request_id
        )
        
        print(f"✅ Predicción completada: {predicted_value:.2f} unidades")
        print(f"📊 Confianza: {final_confidence} ({final_confidence_score:.1%})")
        
        return response
        
    except HTTPException:
        # Re-lanzar HTTPExceptions
        raise
        
    except Exception as e:
        error_msg = f"Error interno del servidor: {str(e)}"
        error_trace = traceback.format_exc()
        print(f"❌ Error en /predict (ID: {request_id}): {error_msg}")
        print(f"🔍 Traceback: {error_trace[:500]}")
        
        # Devolver respuesta de error
        return PredictionResponse(
            predicted_demand=100.0,
            product_category=request.product_category,
            confidence="baja",
            timestamp=datetime.now().isoformat(),
            status="error",
            model_used="simulation",
            confidence_score=0.1,
            message=f"Error: {str(e)[:150]}",
            request_id=request_id
        )

@app.post("/metrics", response_model=BusinessMetricsResponse)
async def calculate_business_metrics(request: BusinessMetricsRequest):
    """
    Calcula métricas de negocio basadas en demandas reales vs predichas.
    
    Útil para evaluar el rendimiento del modelo en producción.
    """
    try:
        if not PREDICTOR_AVAILABLE:
            raise HTTPException(
                status_code=503, 
                detail="Módulo de métricas no disponible"
            )
        
        calculator = BusinessMetrics()
        
        if len(request.actual_demand) != len(request.predicted_demand):
            raise HTTPException(
                status_code=400,
                detail="Las listas actual_demand y predicted_demand deben tener la misma longitud"
            )
        
        if len(request.actual_demand) == 0:
            raise HTTPException(
                status_code=400,
                detail="Las listas no pueden estar vacías"
            )
        
        results = calculator.calculate_forecast_impact(
            request.actual_demand, 
            request.predicted_demand
        )
        
        return BusinessMetricsResponse(**results)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error en /metrics: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error calculando métricas: {str(e)}"
        )

@app.get("/status")
async def system_status():
    """Estado detallado del sistema"""
    return {
        "api": "running",
        "predictor": "loaded" if predictor and predictor.model else "simulation",
        "mlflow": "available" if mlflow_tracker else "unavailable",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.1"
    }

# === EJECUCIÓN LOCAL ===
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 INICIANDO API OLIST FORECASTING - VERSIÓN 2.0.1")
    print("=" * 60)
    
    # Información del sistema
    print(f"📦 Predictor: {'✅ Cargado' if predictor and predictor.model else '⚠️ Simulación'}")
    print(f"📊 MLflow: {'✅ Disponible' if mlflow_tracker else '⚠️ No disponible'}")
    print(f"🔗 URL: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print("=" * 60)
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        print(f"❌ Error iniciando servidor: {e}")
        print("💡 Posibles soluciones:")
        print("1. Verifica que el puerto 8000 no esté en uso")
        print("2. Revisa que todas las dependencias estén instaladas")
        print("3. Ejecuta como administrador si hay problemas de permisos")