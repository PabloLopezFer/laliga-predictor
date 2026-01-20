# 🚀 Guía de Despliegue Gratuito - LaLiga Predictor

Esta guía te explica cómo deployear tu aplicación de forma **GRATUITA** y **SEGURA** sin exponer tus API keys.

## 📋 Requisitos Previos

- Cuenta en GitHub
- API Keys de Football-Data.org y Claude (ya las tienes)

---

## 🔐 PASO 1: Preparar el Proyecto para GitHub (SEGURO)

### 1.1 Crear archivo `.env` (NO se sube a GitHub)

Crea un archivo `.env` en la raíz del proyecto con tus keys REALES:

```env
API_KEY=tu_clave_football_data_aqui
CLAUDE_API_KEY=tu_clave_claude_aqui
PORT=8001
```

### 1.2 Verificar que `.gitignore` incluye `.env`

Ya tienes este archivo creado. Verifica que contenga:

```
.env
__pycache__/
*.pyc
```

### 1.3 El archivo `.env.example` YA ESTÁ LISTO

Este archivo SÍ se sube a GitHub y muestra el formato sin keys reales.

---

## 🌐 PASO 2: Elegir Plataforma de Deploy GRATUITO

Tienes varias opciones gratuitas excelentes:

### ⭐ **OPCIÓN 1: RENDER (RECOMENDADA)**
- ✅ Plan gratuito permanente
- ✅ Python nativo
- ✅ Fácil configuración
- ✅ 750 horas gratis/mes
- ❌ Se duerme tras 15 min inactividad (se despierta en ~30 seg)

### **OPCIÓN 2: Railway**
- ✅ $5 USD gratis/mes
- ✅ Muy rápido
- ✅ No se duerme
- ❌ Créditos limitados

### **OPCIÓN 3: Vercel (con adaptaciones)**
- ✅ Serverless
- ✅ Muy rápido
- ❌ Requiere adaptar el código (más complejo)

---

## 🚀 PASO 3: Deploy en RENDER (Método Recomendado)

### 3.1 Subir a GitHub

1. **Inicializa Git en tu proyecto:**
```bash
cd c:\Users\Pablo\OneDrive\Escritorio\laliga-predictor
git init
git add .
git commit -m "Initial commit - LaLiga Predictor"
```

2. **Crea un repositorio en GitHub:**
   - Ve a https://github.com/new
   - Nombre: `laliga-predictor`
   - Privado o Público (TU ELIGES)
   - NO inicialices con README (ya tienes uno)

3. **Sube el código:**
```bash
git remote add origin https://github.com/TU_USUARIO/laliga-predictor.git
git branch -M main
git push -u origin main
```

⚠️ **IMPORTANTE**: El archivo `.env` NO se subirá gracias al `.gitignore`

### 3.2 Crear Web Service en Render

1. **Ve a https://render.com** y regístrate (gratis)

2. **Clic en "New +" → "Web Service"**

3. **Conecta tu repositorio de GitHub**
   - Autoriza a Render a acceder a GitHub
   - Selecciona `laliga-predictor`

4. **Configuración del servicio:**
   - **Name**: `laliga-predictor` (o el que quieras)
   - **Region**: Frankfurt (más cerca de España)
   - **Branch**: `main`
   - **Root Directory**: (déjalo vacío)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python server.py`
   - **Plan**: **FREE** (gratis)

5. **Variables de Entorno (MUY IMPORTANTE):**
   - Clic en "Advanced"
   - Añade estas variables:
     ```
     API_KEY = tu_clave_football_data_aqui
     CLAUDE_API_KEY = tu_clave_claude_aqui
     PORT = 8001
     ```

6. **Clic en "Create Web Service"**

7. **Espera 2-5 minutos**
   - Render instalará dependencias y arrancará tu app
   - Te dará una URL tipo: `https://laliga-predictor.onrender.com`

### 3.3 ✅ ¡LISTO!

Tu app estará en: `https://tu-nombre.onrender.com`

---

## 🔧 PASO 4: Actualizar `server.py` para Variables de Entorno

El servidor ya debe leer las variables de entorno en lugar de tenerlas hardcodeadas.

**Modifica estas líneas en `server.py`:**

```python
import os

# Leer de variables de entorno (funciona local Y en producción)
API_KEY = os.getenv('API_KEY', 'clave_por_defecto_si_falta')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY', 'clave_por_defecto_si_falta')
PORT = int(os.getenv('PORT', 8001))
```

---

## 📝 PASO 5: Actualizar y Re-deployar

Cada vez que hagas cambios:

```bash
git add .
git commit -m "Descripción de tus cambios"
git push
```

Render detectará los cambios automáticamente y re-deployrá.

---

## 🆓 Alternativa: Railway (Más Rápido pero Créditos Limitados)

### Railway Setup:

1. **Ve a https://railway.app** y regístrate

2. **"New Project" → "Deploy from GitHub repo"**

3. **Selecciona tu repositorio**

4. **Añade Variables:**
   ```
   API_KEY = tu_clave
   CLAUDE_API_KEY = tu_clave
   PORT = 8001
   ```

5. **Railway auto-detecta Python** y lo deployrá

6. **En "Settings" → "Networking"** genera un dominio público

**Ventaja**: No se duerme, más rápido
**Desventaja**: Solo $5 gratis/mes (se puede acabar)

---

## ⚡ Opciones Avanzadas

### Dominio Personalizado (Gratis con Render)

1. **Compra un dominio** (ej: en Namecheap, ~$3/año con `.xyz`)
2. **En Render → Settings → Custom Domain**
3. **Añade tu dominio** y configura DNS según instrucciones

### Mantener Render Despierto

Render se duerme tras 15 min. Opciones:

1. **UptimeRobot** (gratis): Hace ping cada 5 min para mantenerlo despierto
2. **Cron-Job.org**: Similar a UptimeRobot

---

## 🔍 Verificar que Todo Funciona

1. **URL de tu app**: `https://tu-app.onrender.com`
2. **Prueba**: Abre y navega para verificar
3. **Logs**: En Render → Logs para ver errores

---

## ❓ Preguntas Frecuentes

**P: ¿Mis API keys están seguras?**
R: Sí, están en variables de entorno, NO en el código.

**P: ¿Cuánto cuesta?**
R: **GRATIS** con Render o Railway plan free.

**P: ¿Se puede usar en producción?**
R: Sí, para proyectos personales/pequeños es perfecto.

**P: ¿Qué pasa si se acaban los créditos de Railway?**
R: Cambia a Render (gratis ilimitado) o espera al mes siguiente.

**P: ¿Puedo tener dominio propio?**
R: Sí, con Render puedes añadir dominio personalizado gratis.

---

## 📞 Soporte

Si tienes problemas durante el deploy, revisa:

1. **Logs de Render**: Para ver errores
2. **Variables de entorno**: Verifica que estén configuradas
3. **requirements.txt**: Debe estar completo

¡Buena suerte con tu deploy! 🚀
