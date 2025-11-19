#!/usr/bin/env python3
"""
Script para ejecutar actualización y reentrenamiento
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from retraining_scheduler import RetrainingScheduler
import argparse

def main():
    parser = argparse.ArgumentParser(description='Sistema de reentrenamiento automático')
    parser.add_argument('--mode', choices=['once', 'schedule'], default='once',
                       help='Modo de ejecución: once (una vez) o schedule (programado)')
    parser.add_argument('--months', type=int, default=1,
                       help='Número de meses nuevos a simular')
    parser.add_argument('--interval', type=int, default=30,
                       help='Intervalo en días para reentrenamiento programado')
    
    args = parser.parse_args()
    
    scheduler = RetrainingScheduler()
    
    if args.mode == 'once':
        print(f"🔄 Ejecutando reentrenamiento único con {args.months} mes(es) nuevo(s)")
        # Ejecutar una sola vez
        scheduler.monthly_retraining_job()
    else:
        print(f"⏰ Iniciando reentrenamiento programado cada {args.interval} días")
        # Iniciar scheduling
        scheduler.start_scheduler(interval_days=args.interval)

if __name__ == "__main__":
    main()