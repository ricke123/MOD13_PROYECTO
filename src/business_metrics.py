"""
Métricas de negocio para el sistema de forecasting
"""
import numpy as np
import pandas as pd

class BusinessMetrics:
    def __init__(self):
        self.cost_metrics = {
            'inventory_holding_cost': 5.0,      # USD por unidad en inventario
            'stockout_cost': 15.0,              # USD por venta perdida
            'shipping_cost_per_unit': 8.0,      # USD por envío
            'profit_margin_per_unit': 25.0      # USD margen por venta
        }
    
    def calculate_forecast_impact(self, actual_demand, predicted_demand, product_categories=None):
        """
        Calcula el impacto financiero del forecasting
        
        Args:
            actual_demand: array de demanda real
            predicted_demand: array de demanda predicha
            product_categories: array de categorías (opcional)
        """
        actual = np.array(actual_demand)
        predicted = np.array(predicted_demand)
        
        # Evitar división por cero
        actual = np.where(actual == 0, 0.1, actual)
        
        # Métricas de precisión
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        bias = np.mean(predicted - actual)
        
        # Situaciones de negocio
        overstock = np.sum(np.maximum(predicted - actual, 0))  # Exceso de inventario
        stockouts = np.sum(np.maximum(actual - predicted, 0))  # Faltante de stock
        
        # Cálculo de costos
        inventory_costs = overstock * self.cost_metrics['inventory_holding_cost']
        stockout_costs = stockouts * self.cost_metrics['stockout_cost']
        total_costs = inventory_costs + stockout_costs
        
        # Beneficios (ventas que se lograron gracias a mejor forecasting)
        optimal_scenario = np.minimum(actual, predicted)  # Ventas realizadas
        revenue_achieved = optimal_scenario * self.cost_metrics['profit_margin_per_unit']
        total_revenue = np.sum(revenue_achieved)
        
        # Métricas de ROI
        baseline_mape = 25.0  # Asumimos 25% de error en sistema anterior
        improvement = baseline_mape - mape
        roi_percentage = (improvement / baseline_mape) * 100 if baseline_mape > 0 else 0
        
        return {
            'accuracy_percentage': 100 - mape,
            'bias_units': bias,
            'overstock_units': overstock,
            'stockout_units': stockouts,
            'inventory_costs_usd': inventory_costs,
            'stockout_costs_usd': stockout_costs,
            'total_costs_usd': total_costs,
            'estimated_revenue_usd': total_revenue,
            'roi_percentage': roi_percentage,
            'monthly_savings_estimate': total_costs * 30 / len(actual),  # Proyección
            'improvement_vs_baseline': improvement
        }
    
    def calculate_category_breakdown(self, actual_demand, predicted_demand, categories):
        """Análisis detallado por categoría de producto"""
        df = pd.DataFrame({
            'actual': actual_demand,
            'predicted': predicted_demand,
            'category': categories
        })
        
        results = {}
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            if len(cat_data) > 0:
                cat_impact = self.calculate_forecast_impact(
                    cat_data['actual'].values,
                    cat_data['predicted'].values
                )
                results[category] = {
                    'accuracy': cat_impact['accuracy_percentage'],
                    'sample_size': len(cat_data),
                    'total_impact_usd': cat_impact['total_costs_usd'],
                    'avg_demand': cat_data['actual'].mean()
                }
        
        return results

# Función de conveniencia para uso rápido
def quick_business_report(actual, predicted):
    """Reporte rápido de métricas de negocio"""
    calculator = BusinessMetrics()
    return calculator.calculate_forecast_impact(actual, predicted)

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de ejemplo
    actual = [100, 120, 110, 130, 125, 140, 135, 128]
    predicted = [105, 115, 115, 135, 120, 145, 130, 125]
    categories = ['electronics', 'electronics', 'home_appliances', 'electronics', 
                  'home_appliances', 'furniture', 'electronics', 'furniture']
    
    calculator = BusinessMetrics()
    results = calculator.calculate_forecast_impact(actual, predicted)
    
    print("📊 REPORTE DE MÉTRICAS DE NEGOCIO:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    # Análisis por categoría
    category_results = calculator.calculate_category_breakdown(actual, predicted, categories)
    print("\n📈 POR CATEGORÍA:")
    for category, metrics in category_results.items():
        print(f"  {category}: {metrics['accuracy']:.1f}% precisión")


