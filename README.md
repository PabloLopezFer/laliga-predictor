# ⚽ LaLiga Predictor PRO

Sistema de análisis y predicción de partidos de LaLiga usando Inteligencia Artificial con datos reales en tiempo real.

## 🎯 Características

- 🤖 **Análisis con IA** - Predicciones con Claude AI usando datos estadísticos reales
- 📊 **Datos en tiempo real** - API de football-data.org con estadísticas actualizadas
- 📈 **Modelo Avanzado** - Poisson + Dixon-Coles + Regresión Logística Multinomial
- 🏠 **Rendimiento local/visitante** - Estadísticas específicas separadas por ubicación
- 🎲 **Probabilidades 1X2** - Victoria local, empate, victoria visitante (ajustadas con ML)
- ⚽ **Probabilidades de goles** - Over 0.5, 1.5, 2.5, 3.5 y ambos marcan (Poisson)
- 🎯 **xG Ponderado** - 50% temporada + 20% L5 específico + 20% L5 general + 10% L10
- 🔄 **Actualización automática** - Se actualiza cada jornada automáticamente

## 🚀 Uso Local

### 1. Ejecutar el servidor
```bash
python server_proxy.py
```

### 2. Abrir en el navegador
```
http://localhost:8000
```

## 📦 Archivos del Proyecto

- `index.html` - Aplicación web (React + Tailwind)
- `server_proxy.py` - Servidor backend con Python
- `server_new.py` - Servidor alternativo (puerto 8001)

## 🛠️ Tecnologías

- **Frontend**: React 18, Tailwind CSS
- **Backend**: Python (HTTP Server)
- **APIs**: 
  - football-data.org (datos de LaLiga)
  - Anthropic Claude (análisis IA)

## 🌐 Despliegue en la Web

Para desplegar este proyecto en internet, consulta el archivo **`COMO_DESPLEGAR.md`** que contiene:
- Guía paso a paso para Render.com (gratis)
- Cómo configurar las API keys de forma segura
- Instrucciones completas para tener tu web online 24/7

## 📝 Notas

- Las API keys están configuradas en el código para desarrollo local
- Para producción, usa variables de entorno (ver `COMO_DESPLEGAR.md`)
- La web se actualiza automáticamente cada vez que finaliza una jornada

## 👤 Autor

Pablo López - 2026
