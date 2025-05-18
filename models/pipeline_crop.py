import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import streamlit as st

def crop_image(img_bytes, x1, y1, x2, y2):
  """
  Recorta una imagen dada las coordenadas del cuadro delimitador.

  Args:
      image_path: Ruta a la imagen de entrada.
      x1: Coordenada x de la esquina superior izquierda.
      y1: Coordenada y de la esquina superior izquierda.
      x2: Coordenada x de la esquina inferior derecha.
      y2: Coordenada y de la esquina inferior derecha.

  Returns:
      La imagen recortada como un array de NumPy.
  """
  # Leer la imagen
  nparr = np.frombuffer(img_bytes, np.uint8)
  image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

  if image is None:
    raise ValueError("No se pudo decodificar la imagen desde bytes.")

  # Convertir a RGB
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Recortar la imagen
  cropped_image = image_rgb[y1:y2, x1:x2]

  return cropped_image

def detect_objects(img_bytes, path_modelo):
    """
    Detect objects in an image using YOLOv8.

    Args:
        image_path: Path to the input image

    Returns:
        Detected objects and class labels.
    """
    # Load YOLO model
    model = YOLO(path_modelo) # Load the model

    # Leer la imagen
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("No se pudo decodificar la imagen desde bytes.")

    # Convertir a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(image_rgb)[0]

    # Create a copy of the image for drawing
    annotated_image = image_rgb.copy()

    # Generate random colors for classes
    np.random.seed(42)  # For consistent colors
    colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

    # To hold class names and their corresponding colors
    class_labels = {}

    # Process detections
    boxes = results.boxes

    return boxes, results.names, annotated_image, colors

def show_results(img_bytes, confidence_threshold,path_modelo):
    """
    Show original image and detection results side by side.

    Args:
        image_path: Path to the input image
        confidence_threshold: Minimum confidence score for detections
    """

    # Leer la imagen
    nparr = np.frombuffer(img_bytes, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if original_image is None:
        raise ValueError("No se pudo decodificar la imagen desde bytes.")

    # Convertir a RGB
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Get detection results
    boxes, class_names, annotated_image, colors = detect_objects(img_bytes,path_modelo)

    # Process each detected object and apply confidence threshold filtering
    class_labels = {}
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get confidence score
        confidence = float(box.conf[0])

        # Only show detections above confidence threshold
        if confidence > confidence_threshold:
            # Get class id and name
            class_id = int(box.cls[0])
            class_name = class_names[class_id]

            # Get color for this class
            color = colors[class_id % len(colors)].tolist()

            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

            # Store class name and color for legend
            class_labels[class_name] = color

    return boxes

# prompt: dame las funciones de las cajas que se intersequen

# prompt: dame las funciones de las cajas que se intersequen

def iou(boxA, boxB):
  """
  Calcula la Intersección sobre Unión (IoU) de dos cuadros delimitadores.

  Args:
      boxA: Primer cuadro delimitador en formato [x1, y1, x2, y2].
      boxB: Segundo cuadro delimitador en formato [x1, y1, x2, y2].

  Returns:
      El valor IoU.
  """
  # Determinar las coordenadas del rectángulo de intersección
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  # Calcular el área de intersección
  interArea = max(0, xB - xA) * max(0, yB - yA)

  # Calcular el área de ambos cuadros delimitadores
  boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

  # Calcular la Intersección sobre Unión dividiendo el área de intersección
  # por la suma de las áreas de los cuadros delimitadores menos el área de intersección
  iou = interArea / float(boxAArea + boxBArea - interArea)

  return iou


def get_intersecting_boxes_and_functions(boxes, iou_threshold=0.5):
    """
    Identifica los cuadros delimitadores que se intersecan y sus funciones asociadas.

    Args:
        boxes: Una lista de cuadros delimitadores en formato [x1, y1, x2, y2].
               Se asume que cada cuadro delimitador corresponde a una función.
               La lista 'boxes' debería ser la salida de la función show_results.
        iou_threshold: El umbral de IoU para considerar que dos cajas se intersecan.

    Returns:
        Una lista de tuplas, donde cada tupla contiene un par de índices de cuadros
        delimitadores que se intersecan. Por ejemplo, [(0, 1), (2, 3)].
        También devuelve una lista de las funciones asociadas a los cuadros
        que se intersecan (en este ejemplo, serían las funciones correspondientes
        a los índices 0, 1, 2 y 3 en la lista de cajas original).
    """
    if(len(boxes) == 0):
      return ([],[])

    intersecting_pairs = []
    intersecting_functions = []

    # Iterar sobre todas las combinaciones de cuadros delimitadores
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            boxA = boxes[i].xyxy[0]  # Obtener coordenadas [x1, y1, x2, y2] del tensor
            boxB = boxes[j].xyxy[0]

            # Calcular el IoU entre los dos cuadros
            overlap = iou(boxA, boxB)

            # Si el IoU es mayor que el umbral, se consideran intersectantes
            if overlap > iou_threshold:
                intersecting_pairs.append((i, j))
                intersecting_functions.append((boxes[i].conf[0].item(), boxes[j].conf[0].item()))



    return intersecting_pairs, intersecting_functions


def nodos_eliminados(intersecciones, valores):

  # 1. Construimos el grafo de conflictos
  grafo = defaultdict(set)
  for (a, b) in intersecciones:
      grafo[a].add(b)
      grafo[b].add(a)

  # 2. Calculamos el valor máximo por nodo
  valores_por_nodo = defaultdict(float)
  for (par, val) in zip(intersecciones, valores):
      a, b = par
      v = max(val)
      valores_por_nodo[a] = max(valores_por_nodo[a], v)
      valores_por_nodo[b] = max(valores_por_nodo[b], v)

  # 3. Algoritmo greedy: elegir nodo con mayor valor, eliminarlo y sus vecinos
  nodos_disponibles = set(valores_por_nodo.keys())
  resultado = set()

  while nodos_disponibles:
      # Escoge el nodo con mayor valor
      mejor = max(nodos_disponibles, key=lambda n: valores_por_nodo[n])
      resultado.add(mejor)

      # Elimina el nodo y sus vecinos del conjunto
      nodos_disponibles -= {mejor}
      nodos_disponibles -= grafo[mejor]

  todos_los_nodos = set()
  for a, b in intersecciones:
      todos_los_nodos.add(a)
      todos_los_nodos.add(b)

  nodos_a_eliminar = sorted(todos_los_nodos - resultado)
  return nodos_a_eliminar


def limpieza_rectangulos(boxes, confidence_threshold):
  """
  Elimina los cuadros delimitadores que se intersecan y sus funciones asociadas.

  Args:
      boxes: Una lista de cuadros delimitadores en formato [x1, y1, x2, y2].
      confidence : El umbral de confianza para eliminar las funciones.
  """
  # Obtener las intersecciones y las funciones asociadas
  intersecciones, funciones = get_intersecting_boxes_and_functions(boxes,confidence_threshold)
  nodos_a_eliminar = nodos_eliminados(intersecciones, funciones)

  mascara = torch.ones(len(boxes), dtype=torch.bool)
  mascara[nodos_a_eliminar] = False  # Eliminar esos nodos

  # Filtrar los datos internos del objeto boxes
  boxes.data = boxes.data[mascara]

  return boxes


def proceso_general(img_bytes, path_modelo, confidence1 = 0.01, confidence2 = 0.1):
  """
  Hace el pipeline completo de procesamiento de imagen.

  Args:
      image_path: Ruta a la imagen de entrada.
      confidence_threshold: Umbral de confianza para la detección.
      model: El modelo YOLOv8 cargado para la detección.

  Returns:
      Una lista de imágenes recortadas resultantes de la limpieza.
  """
  boxes = show_results(img_bytes, confidence1, path_modelo)
  boxes = limpieza_rectangulos(boxes, confidence2)
  images = []
  coordenadas = []

  x_min_total = float('inf')
  y_min_total = float('inf')
  x_max_total = float('-inf')
  y_max_total = float('-inf')

  for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    croppeded_image = crop_image(img_bytes, x1, y1, x2, y2)
    coordenadas.append((x1, y1, x2, y2))
    images.append(croppeded_image)
    x_min_total = min(x_min_total, x1,x2)
    y_min_total = min(y_min_total, y1,y2)
    x_max_total = max(x_max_total, x1,x2)
    y_max_total = max(y_max_total, y1,y2)


  rectangulo_grande = [x_min_total,y_min_total,x_max_total,y_max_total]
  data = {'boxes': coordenadas, 'images': images}


  return data, rectangulo_grande

def asignar_posiciones_en_cuadricula(rectangulos, tolerancia=600):
    # Cada rectángulo es (x_max, y_min, x_min, y_max)

    # Paso 1: Ordenar por y_max de mayor a menor (más arriba primero)
    rectangulos_ordenados = sorted(rectangulos, key=lambda r: -r[3])  # r[3] = y_max

    filas = []

    for rect in rectangulos_ordenados:
        agregado = False
        for fila in filas:
            # Comparar con la primera figura de la fila (por y_max)
            if abs(fila[0][3] - rect[3]) <= tolerancia:
                fila.append(rect)
                agregado = True
                break
        if not agregado:
            filas.append([rect])

    # Paso 2: Ordenar cada fila por x_min (más a la izquierda primero)
    for fila in filas:
        fila.sort(key=lambda r: r[2])  # r[2] = x_min

    # Paso 3: Asignar coordenadas (fila, columna)
    posicion_por_rect = {}
    for i, fila in enumerate(filas):
        for j, rect in enumerate(fila):
            posicion_por_rect[tuple(rect)] = (i + 1, j + 1)

    return posicion_por_rect

def rectangulos_intersectan(r1, r2):
    x1_max, y1_min, x1_min, y1_max = r1
    x2_min, y2_min, x2_max, y2_max = r2

    # No hay intersección si uno está completamente a la izquierda/derecha o arriba/abajo del otro
    if x1_min >= x2_max or x2_min >= x1_max:
        return False
    if y1_min >= y2_max or y2_min >= y1_max:
        return False
    return True

def encontrar_espacios_vacios_filas(rectangulos, posiciones, rectangulo_grande, tolerancia=50):
    from collections import defaultdict

    # Agrupar rectángulos por fila
    filas = defaultdict(list)
    for rect, (fila, _) in posiciones.items():
        filas[fila].append(rect)

    espacios_vacios = []

    for fila_id in sorted(filas):
        fila = filas[fila_id]

        for i in range(len(fila) - 1):
            rect_izq = fila[i]
            rect_der = fila[i + 1]

            # Definir límites del espacio entre rectángulos
            espacio_x_min = max(rect_izq[0], rect_izq[2]) # x_max del rectángulo izquierdo
            espacio_x_max = min(rect_der[2], rect_der[2])  # x_min del rectángulo derecho
            ancho = espacio_x_max - espacio_x_min
            y_min = max(rect_izq[1], rect_der[1])
            y_max = min(rect_izq[3], rect_der[3])
            if ancho >= tolerancia:
                nuevo_espacio = (espacio_x_max, y_min, espacio_x_min, y_max)

                # Verifica que no intersecte con ningún rectángulo existente
                interseca = any(rectangulos_intersectan(nuevo_espacio, r) for r in rectangulos)

                if not interseca:
                    espacios_vacios.append(nuevo_espacio)



    return espacios_vacios


def toda_la_info(image_path):
    """
    Hace el pipeline completo de procesamiento de imagen.

    Args:
      image_path: Ruta a la imagen de entrada.
      confidence_threshold: Umbral de confianza para 
      la detección.
      model: El modelo YOLOv8 cargado para la detección.

    Returns:
      Una lista de imágenes recortadas resultantes de la limpieza.
      data es el box y la imagen croppeada del box en matriz rgb por pixel. coord es box e img es rgb
      rect grande es todo el anaquel
      posicion por rect es la coordenada del cuadrito. si m es 5 y n es 4, un ejemplo sería (1,2) de la matrix m x n
    """
    data, rectangulo_grande = proceso_general(image_path, "models\\best.pt", confidence1 = 0.01, confidence2 = 0.1)
    posicion_por_rect = asignar_posiciones_en_cuadricula(data["boxes"],rectangulo_grande[2]/6)
    for i in list(posicion_por_rect):
        if posicion_por_rect[i][0] > 4:
            del posicion_por_rect[i]

    dict_vacios = {}
    espacios_vacios = encontrar_espacios_vacios_filas(data["boxes"], posicion_por_rect, rectangulo_grande, tolerancia=50)
    x = 1
    for i in espacios_vacios:
        dict_vacios[i] = x
        x += 1

    return [data, rectangulo_grande, posicion_por_rect, espacios_vacios, dict_vacios]
def visualizar(img_bytes):
    """
    Show original image and detection results side by side.

    Args:
        image_path: Path to the input image
        confidence_threshold: Minimum confidence score for detections
    """
    # Leer la imagen
    nparr = np.frombuffer(img_bytes, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if original_image is None:
        raise ValueError("No se pudo decodificar la imagen desde bytes.")

    # Convertir a RGB
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    annotated_image = original_image.copy()

    data, rectangulo_grande = proceso_general(img_bytes, "models\\best.pt", confidence1 = 0.01, confidence2 = 0.1)
    posicion_por_rect = asignar_posiciones_en_cuadricula(data["boxes"])
    espacios_vacios = encontrar_espacios_vacios_filas(data["boxes"], posicion_por_rect, rectangulo_grande, tolerancia=10)


    # Process each detected object and apply confidence threshold filtering
    class_labels = {}
    for box in data["boxes"]:
        # Draw bounding box
        cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), (0,0,0), 2)
        label = str(posicion_por_rect[box])  # starts from 1
        position = (box[0], box[1])  # slightly above the top-left corner
        cv2.putText(annotated_image, label, position, cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 3)

    for box in espacios_vacios:
      cv2.rectangle(annotated_image, (box[2], box[1]), (box[0], box[3]), (255, 255, 255), -1)

    # Create figure
    plt.figure(figsize=(15, 7))

    # Show original image
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    plt.axis('off')

    # Show detection results
    plt.subplot(1, 2, 2)
    plt.title('Detected Objects')
    plt.imshow(annotated_image)
    plt.axis('off')

    plt.tight_layout()
    st.pyplot(plt)
    plt.show()
