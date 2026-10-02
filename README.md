# Personal Video Stock

Aplicación personal para buscar, seleccionar, recortar y editar footage de video.

## Requisitos (Windows)

1. **Python 3.10 o superior**  
   https://www.python.org/downloads/  
   ⚠️ Al instalar marca la casilla **"Add Python to PATH"**

2. **FFmpeg** (necesario para recortar y guardar clips)  
   Forma más fácil en PowerShell/CMD:
   ```
   winget install ffmpeg
   ```
   O descarga desde: https://ffmpeg.org/download.html

## Cómo arrancar (Windows)

### Opción fácil
Haz **doble clic** en `start.bat`

### Opción manual
Abre **CMD** o **PowerShell** en la carpeta del proyecto:

```cmd
pip install -r requirements.txt
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Luego abre en el navegador:

| Página | URL |
|--------|-----|
| Búsqueda de footage | http://localhost:8000 |
| Revisar y guardar clip | http://localhost:8000/review |
| Clip Edition (recorte iMovie) | http://localhost:8000/editor |
| Video Creation (montaje IA) | http://localhost:8000/create |

## API (v0.3)

| Método | Ruta | Qué hace |
|--------|------|----------|
| POST | `/api/search` | Busca footage (`platform`: youtube, dailymotion, tiktok, instagram, all) |
| POST | `/api/save-clip` | Descarga, recorta y guarda un clip |
| GET | `/api/clips` | Lista MP4 en `storage/clips` |
| POST | `/api/upload-media` | Sube un video local a `storage/uploads` |
| POST | `/api/export` | Renderiza timeline (clips + textos) a MP4 con FFmpeg |
| POST | `/api/create/plan` | IA arma guion + escenas con clips de `storage/clips` |
| GET | `/api/health` | Estado del servidor y si FFmpeg está disponible |

## Qué hace la app

1. Escribes la descripción del footage que necesitas
2. Busca en YouTube y sugiere momentos (timestamps)
3. Seleccionas el segmento → ajustas inicio/fin
4. Guardas el clip con nombre personalizado (quita subtítulos suaves)
5. Abres el Editor para timeline, textos y exportación

## Estructura

```
video-stock-app/
├── start.bat              ← Doble clic para arrancar (Windows)
├── requirements.txt
├── backend/
│   ├── main.py
│   ├── search_youtube.py
│   ├── clip_processor.py
│   └── video_tools.py
├── frontend/
│   ├── index.html         ← Búsqueda
│   ├── review.html        ← Revisar / guardar
│   └── editor/
│       └── index.html     ← Editor timeline
└── storage/
    ├── clips/             ← Clips guardados y exports
    ├── uploads/           ← Videos locales subidos desde el editor
    ├── references/
    └── temp/
```

## Notas

- La primera vez que corres `start.bat` instala las librerías (puede tardar 1-2 minutos).
- YouTube a veces bloquea IPs de servidores. En tu PC casera normalmente funciona bien.
- Búsqueda: YouTube, Dailymotion, TikTok, Instagram, o **Todas** a la vez.
- Instagram y TikTok son frágiles (sin login); a veces no hay resultados o no se puede descargar el clip. Dailymotion y YouTube suelen ir mejor.
- El editor exporta MP4 de verdad con FFmpeg: recorta clips, los concatena y pinta textos (drawtext).
- Los videos cargados desde el PC se suben a `storage/uploads` para poder renderizarlos.
- Los MP4 exportados quedan en `storage/clips` y se pueden reimportar con **Clips de Stock**.
