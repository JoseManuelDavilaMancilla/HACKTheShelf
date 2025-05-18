import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import os
from streamlit_drawable_canvas import st_canvas
from io import BytesIO

PIXELS_POR_CM = 10
ANAQUEL_ANCHO_CM = 105
PISO_ALTO_CM = 32.02
LINEA_GROSOR_CM = 1
ESPACIO_INFERIOR_CM = 10
ESPACIO_SUPERIOR_CM = 10

ANAQUEL_ANCHO_PX = int(ANAQUEL_ANCHO_CM * PIXELS_POR_CM)
PISO_ALTO_PX = int(PISO_ALTO_CM * PIXELS_POR_CM)
LINEA_GROSOR_PX = int(LINEA_GROSOR_CM * PIXELS_POR_CM)
ESPACIO_INFERIOR_PX = int(ESPACIO_INFERIOR_CM * PIXELS_POR_CM)
ESPACIO_SUPERIOR_PX = int(ESPACIO_SUPERIOR_CM * PIXELS_POR_CM)

st.set_page_config(layout="wide")
st.title("🧠 Planograma Interactivo con Clics")

csv_file = st.file_uploader("Sube tu archivo CSV", type="csv")
img_folder = st.text_input("Ruta a la carpeta de imágenes", "models/HackFEMSA")

@st.cache_data
def cargar_datos(csv_path):
    df = pd.read_csv(csv_path, encoding="latin1")
    for col in ['Altura', 'Ancho']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.replace('cm', '', regex=False).str.strip().astype(float)
    return df

def generar_anaquel(df, anaquel, img_folder):
    subset = df[df['Anaquel'] == anaquel]
    max_charola = subset['Charola'].max()
    img_height = ESPACIO_SUPERIOR_PX + max_charola * (PISO_ALTO_PX + LINEA_GROSOR_PX) + ESPACIO_INFERIOR_PX
    canvas = Image.new("RGB", (ANAQUEL_ANCHO_PX, img_height), "white")
    draw = ImageDraw.Draw(canvas)
    boxes = []

    for charola in sorted(subset['Charola'].unique()):
        fila = subset[subset['Charola'] == charola]
        x_cursor = 0

        for _, row in fila.iterrows():
            try:
                img_path = os.path.join(img_folder, f"{row['Nombre']}.png")
                img = Image.open(img_path).convert("RGBA")

                w_px = int(row["Ancho"] * PIXELS_POR_CM)
                h_px = int(row["Altura"] * PIXELS_POR_CM)
                cantidad = int(row["Cantidad de Frentes"])
                img_resized = img.resize((w_px, h_px))
                nivel_base = img_height - ESPACIO_INFERIOR_PX - (charola * (PISO_ALTO_PX + LINEA_GROSOR_PX))

                if row["Ancho"] > row["Altura"]:
                    for i in range(cantidad):
                        y_pos = nivel_base + PISO_ALTO_PX - ((i + 1) * h_px + i * 2)
                        if y_pos < ESPACIO_SUPERIOR_PX:
                            break
                        canvas.paste(img_resized, (x_cursor, y_pos), img_resized)
                        boxes.append({"x": x_cursor, "y": y_pos, "w": w_px, "h": h_px,
                                       "Nombre": row['Nombre'], "Inventario": row['Inventario'], "Precio": row['Precio']})
                else:
                    for i in range(cantidad):
                        x_pos = x_cursor + i * (w_px + 2)
                        y_pos = nivel_base + (PISO_ALTO_PX - h_px)
                        if x_pos + w_px > ANAQUEL_ANCHO_PX:
                            break
                        canvas.paste(img_resized, (x_pos, y_pos), img_resized)
                        boxes.append({"x": x_pos, "y": y_pos, "w": w_px, "h": h_px,
                                       "Nombre": row['Nombre'], "Inventario": row['Inventario'], "Precio": row['Precio']})
                x_cursor += cantidad * (w_px + 2) + 5

            except FileNotFoundError:
                continue

        linea_y = img_height - ESPACIO_INFERIOR_PX - ((charola - 1) * (PISO_ALTO_PX + LINEA_GROSOR_PX)) - LINEA_GROSOR_PX
        draw.rectangle([(0, linea_y), (ANAQUEL_ANCHO_PX, linea_y + LINEA_GROSOR_PX)], fill="black")

    return canvas, boxes

if csv_file:
    df = cargar_datos(csv_file)
    df['Fila'] = df['Anaquel'].astype(str).str.extract(r'(F\d+)')
    df['Lado'] = df['Anaquel'].astype(str).str.extract(r'F\d+([AB])')
    df['Grupo'] = df['Fila'] + df['Lado']

    grupos_disponibles = sorted(df['Grupo'].dropna().unique())
    modo_vista = st.radio("Modo de visualización", ["Ver por fila completa", "Ver anaquel individual"])

    if modo_vista == "Ver por fila completa":
        grupo_sel = st.selectbox("Selecciona grupo (Fila + Lado)", grupos_disponibles)
        anaqueles_grupo = sorted(df[df['Grupo'] == grupo_sel]['Anaquel'].unique())

        imagenes = []
        for anaquel_id in anaqueles_grupo:
            img, _ = generar_anaquel(df, anaquel_id, img_folder)
            if img:
                imagenes.append(img)

        if imagenes:
            total_width = sum(im.width for im in imagenes)
            max_height = max(im.height for im in imagenes)
            canvas_total = Image.new("RGB", (total_width, max_height), "white")
            x_offset = 0
            for img in imagenes:
                canvas_total.paste(img, (x_offset, 0))
                x_offset += img.width
            st.image(canvas_total, caption=f"Planograma combinado - {grupo_sel}", use_column_width=False)

            buffer = BytesIO()
            canvas_total.save(buffer, format="PNG")
            st.download_button("Descargar como PNG", data=buffer.getvalue(), file_name=f"planograma_{grupo_sel}.png", mime="image/png")

    else:
        anaqueles_disponibles = sorted(df['Anaquel'].unique())
        seleccion = st.selectbox("Selecciona un anaquel a visualizar", options=anaqueles_disponibles)

        if seleccion:
            anaquel_img, boxes = generar_anaquel(df, seleccion, img_folder)

            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=0,
                background_image=anaquel_img,
                update_streamlit=True,
                height=anaquel_img.height,
                width=anaquel_img.width,
                drawing_mode="point",
                key="canvas"
            )

            if canvas_result.json_data and canvas_result.json_data.get("objects"):
                last_obj = canvas_result.json_data["objects"][-1]
                cx = last_obj.get("left", 0)
                cy = last_obj.get("top", 0)

                for box in boxes:
                    if box['x'] <= cx <= box['x'] + box['w'] and box['y'] <= cy <= box['y'] + box['h']:
                        st.success("**{}**\n\nInventario: {}\nPrecio: ${}".format(
                            box['Nombre'], box['Inventario'], box['Precio']))
                        break

