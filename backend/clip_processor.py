"""
Lógica para sugerir segmentos relevantes según la descripción del usuario.
Por ahora usa matching simple de keywords + scoring.
Más adelante se puede conectar a un LLM real.
"""

from typing import List, Dict
import re
from difflib import SequenceMatcher


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def suggest_segments(query: str, transcript: List[Dict], max_segments: int = 4) -> List[Dict]:
    """
    Analiza el transcript y sugiere los mejores rangos de tiempo
    que coinciden con la descripción del usuario.
    """
    if not transcript:
        return []

    query_norm = normalize(query)
    query_words = set(query_norm.split())

    # Crear ventanas de ~8-20 segundos
    candidates = []
    window_size = 6  # número de segmentos de transcript por ventana

    for i in range(len(transcript)):
        window = transcript[i:i + window_size]
        if not window:
            continue

        combined_text = " ".join([s["text"] for s in window])
        combined_norm = normalize(combined_text)

        # Score simple
        word_overlap = len(query_words.intersection(set(combined_norm.split())))
        sim = similarity(query_norm, combined_norm)
        
        score = (word_overlap * 0.4) + (sim * 0.6)

        if score > 0.15:  # umbral mínimo
            start = window[0]["start"]
            end = window[-1]["start"] + window[-1]["duration"]
            
            # Evitar clips demasiado cortos o largos
            duration = end - start
            if 3 <= duration <= 45:
                candidates.append({
                    "start": round(start, 1),
                    "end": round(end, 1),
                    "text": combined_text[:180] + ("..." if len(combined_text) > 180 else ""),
                    "confidence": round(min(score, 0.98), 2),
                    "reason": f"Coincidencia de palabras clave y contexto semántico (score {score:.2f})"
                })

    # Ordenar por confianza y eliminar solapamientos fuertes
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    
    selected = []
    for cand in candidates:
        overlap = False
        for sel in selected:
            # Si se solapan mucho, saltar
            if abs(cand["start"] - sel["start"]) < 8:
                overlap = True
                break
        if not overlap:
            selected.append(cand)
        if len(selected) >= max_segments:
            break

    return selected
