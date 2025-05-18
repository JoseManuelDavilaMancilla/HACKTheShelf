# Smart Planogram Helper - Hackathon 2025

Este proyecto fue desarrollado como solución para el reto del hackathon: **"¿Cómo podemos ayudar a los empleados de OXXO a organizar mejor sus anaqueles?"**.

Integra una solución web con modelos de IA para detectar productos en anaqueles, clasificarlos y sugerir automáticamente un planograma óptimo.

---

## Índice

- [Demo](#demo)
- [Tecnologías](#tecnologías)
- [Arquitectura del Proyecto](#arquitectura-del-proyecto)
- [Instalación](#instalación)
- [Módulos](#módulos)
  - [1. Frontend Web](#1-frontend-web)
  - [2. Backend con MongoDB](#2-backend-con-mongodb)
  - [3. Reconocimiento de Objetos (YOLOv3)](#3-reconocimiento-de-objetos-yolov3)
  - [4. Clasificación de Objetos](#4-clasificación-de-objetos)
  - [5. Optimización de Planogramas](#5-optimización-de-planogramas)
- [Licencia](#licencia)

---

## Demo

Próximamente: video demo e imágenes de ejemplo.

---

## Tecnologías

- **Frontend:** Streamlit
- **Backend:** MongoDB Atlas
- **IA Computer Vision:** YOLOv3 (PyTorch)
- **Clasificación de productos:** Clip + Cloudflare
- **Optimización de planogramas:** Algoritmo heurístico tipo greedy
- **Almacenamiento:** Mongo DB (imágenes)

---

## Arquitectura del Proyecto

```
 Usuario
   │
Frontend Web (Streamlit)
   │
Backend API (Streamlit + MongoDB)
   │
┌──────────────┬──────────────────────┬──────────────────────┐
│ Reconocimiento (YOLOv3) │ Clasificación │ Optimización     │
│ Pipeline Mejora         │ Clip + Cloud  │ Planograma ideal │
└──────────────┴──────────────────────┴──────────────────────┘
```

---

## Instalación

```bash
git clone 

```

### Backend

```bash
cd backend
npm install
npm run dev
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Módulos

### 1. Frontend Web

- Upload de imagen del anaquel
- Vista de productos detectados con bounding boxes
- Visualización del planograma sugerido
- Flujo simple para empleados con baja carga cognitiva

### 2. Backend con MongoDB

- API RESTful
- Conexión con modelos de IA
- Almacenamiento de resultados

### 3. Reconocimiento de Objetos (YOLOv3 + cloudflare + clip)

- Entrenado con +2,000 imágenes de anaqueles
- Bounding boxes sin clasificación
- mAP@0.5 = 92%
- Postprocesamiento para mejorar bordes y recortes
- Asignación de objetos a una grilla de `(fila, columna)`

```bash
python detectar_productos.py --imagen ./imagenes/anaquel.jpg
```

### 4. Clasificación de Objetos

- Modelo Hibrido CLip + Cloudflare
- cluster entrenada con ~50 clases genéricas (bebidas, snacks, etc.)
- Asigna etiqueta a cada bounding box
- Accuracy de top-1: 88% sobre validación cruzada
- Ideal para alimentar reglas del planograma

### 5. Optimización de Planogramas

- Algoritmo heurístico basado en:
  - Popularidad del producto
  - Tipo de anaquel y nivel de acceso
  - Cobertura, balance y reglas de merchandising
- Salida en formato JSON y visual

```json
{
  "suggested_planogram": [
    [{"product": "Coca-Cola", "cell": [0,0]}, ...]
  ]
}
```

---

## Licencia

MIT © 2025 - Proyecto desarrollado para fines académicos y demostrativos.  
No afiliado con FEMSA ni OXXO.
