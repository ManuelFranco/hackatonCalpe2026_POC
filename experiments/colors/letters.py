from PIL import Image, ImageDraw, ImageFont

# ===== CONFIGURACIÓN =====
LETRA = "C"

COLOR_1 = "red"
COLOR_2 = "blue"

FONDO = "white"

ANCHO = 500
ALTO = 500
TAMANO_FUENTE = 350

# Fuente de Windows
FUENTE = r"C:\Windows\Fonts\arialbd.ttf"


def crear_imagen(letra, color, nombre_archivo):
    imagen = Image.new("RGB", (ANCHO, ALTO), FONDO)
    dibujo = ImageDraw.Draw(imagen)

    fuente = ImageFont.truetype(FUENTE, TAMANO_FUENTE)

    # Obtener dimensiones de la letra
    caja = dibujo.textbbox((0, 0), letra, font=fuente)

    ancho_texto = caja[2] - caja[0]
    alto_texto = caja[3] - caja[1]

    # Centrar la letra
    x = (ANCHO - ancho_texto) / 2 - caja[0]
    y = (ALTO - alto_texto) / 2 - caja[1]

    dibujo.text(
        (x, y),
        letra,
        font=fuente,
        fill=color
    )

    imagen.save(nombre_archivo)


# ===== CREAR IMÁGENES =====
crear_imagen(LETRA, COLOR_1, "red_" + LETRA + ".png")
crear_imagen(LETRA, COLOR_2, "blue_" + LETRA + ".png")

print("Imágenes creadas correctamente.")