"""
ALGO-ELLIOTT PRO — capa1_funciones.py
========================================
Librería reutilizable con todos los cálculos del bloque puente de la Capa 1
(Sección 0.4 del protocolo). Se guarda en el repositorio del puente de datos
para no tener que reescribir esta lógica en cada chat nuevo.

USO TÍPICO EN UN CHAT NUEVO:
    curl -s .../capa1_funciones.py -o capa1_funciones.py
    (descargar los JSON del puente como siempre, con curl)
    python3
    >>> from capa1_funciones import *
    >>> velas = limpiar_velas(json.load(open('AAPL.json'))['velas'])
    >>> evaluar_candidato(velas_diarias, velas_semanales, velas_benchmark_sem)

Todas las funciones trabajan sobre listas de velas ya cargadas (dict con
claves t,o,h,l,c,v) — el propio script NO descarga nada, eso se sigue
haciendo con curl como hasta ahora. Esto es solo la capa de cálculo.
"""

import json
import numpy as np


# ============================================================
# 1. LIMPIEZA DE DATOS
# ============================================================

def limpiar_velas(velas: list) -> list:
    """Descarta velas con cierre nulo (NaN) — el fallo que casi arruinó
    el escaneo europeo del 10 de agosto de 2026 (Enmienda V15.1, Sección 0.4).
    SIEMPRE aplicar esto antes de cualquier cálculo.
    """
    return [v for v in velas
            if v.get("c") is not None
            and str(v["c"]) != "nan"
            and not (isinstance(v["c"], float) and np.isnan(v["c"]))]


def cierres(velas: list) -> np.ndarray:
    """Atajo: array de precios de cierre de una lista de velas ya limpia."""
    return np.array([v["c"] for v in velas])


def fecha_ultima_vela(velas: list) -> str:
    """Devuelve la fecha (YYYY-MM-DD) de la última vela válida — para
    declarar SIEMPRE la frescura del dato, tal como exige el protocolo."""
    return velas[-1]["t"][:10] if velas else "sin datos"


# ============================================================
# 2. SCORE DE MOMENTUM (Sección 16.3 + Ilmanen 16.3 ponderación)
# ============================================================

def calcular_score(velas_diarias: list) -> dict | None:
    """Score = (Ret1M% + Ret3M%) x Efficiency Ratio(21d).
    Fórmula propia, parámetros NO validados empíricamente — declarado
    así en el protocolo (V15.25).
    """
    v = limpiar_velas(velas_diarias)
    if len(v) < 63:
        return None
    c = cierres(v)
    p = c[-1]
    r1 = float((p / c[-21] - 1) * 100)
    r3 = float((p / c[-63] - 1) * 100)
    tramo = c[-21:]
    er = float(abs(tramo[-1] - tramo[0]) / np.sum(np.abs(np.diff(tramo))))
    return {"precio": p, "ret_1m": r1, "ret_3m": r3, "er": er,
            "score": (r1 + r3) * er, "fecha": fecha_ultima_vela(v)}


# ============================================================
# 3. FROG-IN-THE-PAN — calidad de trayectoria (Gray & Vogel, Cap.6)
# ============================================================

def calcular_fip(velas_diarias: list) -> float | None:
    """ID = signo(retorno) x [%dias negativos - %dias positivos].
    ID BAJO (negativo) = trayectoria CONTINUA = mejor calidad.
    ID ALTO = trayectoria DISCRETA = peor calidad (aunque el score sea alto).
    Independiente del Efficiency Ratio (correlación verificada +0.017).
    """
    v = limpiar_velas(velas_diarias)
    if len(v) < 63:
        return None
    c = cierres(v)
    r = np.diff(c) / c[:-1]
    ret_total = c[-1] / c[0] - 1
    pos = np.sum(r > 0) / len(r) * 100
    neg = np.sum(r < 0) / len(r) * 100
    return float(np.sign(ret_total) * (neg - pos))


def es_trayectoria_continua(id_fip: float, percentil20_universo: float) -> bool:
    """Un candidato es 'continuo' si su ID está en el quintil inferior
    (más negativo) del universo completo que se está escaneando ese día.
    percentil20_universo se calcula UNA VEZ sobre todos los candidatos
    del escaneo, no por candidato individual."""
    return id_fip <= percentil20_universo


# ============================================================
# 4. MOMENTUM DUAL + MOMENTUM FRESCO (Antonacci Cap.9, Chen/Kadan/Kose)
# ============================================================

def calcular_momentum_dual(velas_semanales: list, velas_benchmark_sem: list) -> dict | None:
    """Relativo (vs benchmark, 13 semanas) y Absoluto (12 meses, 53 semanas).
    Absoluto <= 0 -> descartar sin más filtros (regla de Antonacci)."""
    w = limpiar_velas(velas_semanales)
    b = limpiar_velas(velas_benchmark_sem)
    if len(w) < 53 or len(b) < 13:
        return None
    cw = cierres(w)
    cb = cierres(b)
    rel = float(((cw[-1] / cw[-13] - 1) - (cb[-1] / cb[-13] - 1)) * 100)
    absoluto = float((cw[-1] / cw[-53] - 1) * 100)
    return {"relativo": rel, "absoluto": absoluto, "pasa": absoluto > 0}


def es_momentum_fresco(velas_semanales: list) -> bool | None:
    """Ganador fresco (Chen/Kadan/Kose 2009): fuerte en los últimos 12 meses
    PERO débil en los 12-24 meses anteriores. Distingue un valor que acaba
    de girar de otro que lleva 2 años subiendo (posible agotamiento)."""
    w = limpiar_velas(velas_semanales)
    if len(w) < 105:
        return None
    cw = cierres(w)
    m12 = (cw[-1] / cw[-53] - 1) * 100
    m24_12 = (cw[-53] / cw[-105] - 1) * 100
    return bool(m12 > 0 and m24_12 <= 0)


# ============================================================
# 5. ETAPA DE WEINSTEIN (reclasificado a Capa 1 en V15.24)
# ============================================================

def calcular_etapa_weinstein(velas_semanales: list) -> dict | None:
    """Etapa 2 (alcista) = precio sobre su media de 30 semanas Y esa
    media con pendiente positiva (>0.5% en 5 semanas). Solo Etapa 2 pasa
    el filtro de la Capa 1."""
    w = limpiar_velas(velas_semanales)
    if len(w) < 35:
        return None
    cw = cierres(w)
    ma30 = np.convolve(cw, np.ones(30) / 30, mode="valid")
    pendiente = float((ma30[-1] - ma30[-5]) / ma30[-5] * 100)
    sobre_media = bool(cw[-1] > ma30[-1])
    if sobre_media and pendiente > 0.5:
        etapa = "2 ALCISTA"
    elif sobre_media:
        etapa = "3 techo/distribución"
    elif pendiente < -0.5:
        etapa = "4 BAJISTA"
    else:
        etapa = "1 base/acumulación"
    return {"etapa": etapa, "pendiente_ma30": pendiente,
            "pasa_capa1": etapa == "2 ALCISTA"}


# ============================================================
# 6. PROXY KONCORDE — manos fuertes/débiles (NVI/PVI de Fosback)
# ============================================================

def calcular_koncorde_proxy(velas_semanales: list) -> dict | None:
    """Azul (NVI, manos fuertes) vs Verde (PVI, manos débiles), cada uno
    frente a su propia media exponencial de 26 periodos.
    Descarte: azul negativo Y verde positivo a la vez = institucional
    saliendo mientras entra el minorista (patrón de distribución)."""
    w = limpiar_velas(velas_semanales)
    if len(w) < 30:
        return None
    cw = cierres(w)
    vw = np.array([x.get("v", 0) or 0 for x in w])

    def ema(a, n):
        k = 2 / (n + 1)
        out = [a[0]]
        for x in a[1:]:
            out.append(k * x + (1 - k) * out[-1])
        return np.array(out)

    nvi, pvi = [1000.0], [1000.0]
    for i in range(1, len(cw)):
        cambio = (cw[i] - cw[i - 1]) / cw[i - 1]
        if vw[i] < vw[i - 1]:
            nvi.append(nvi[-1] * (1 + cambio)); pvi.append(pvi[-1])
        else:
            pvi.append(pvi[-1] * (1 + cambio)); nvi.append(nvi[-1])
    nvi, pvi = np.array(nvi), np.array(pvi)
    azul = float((nvi[-1] / ema(nvi, 26)[-1] - 1) * 100)
    verde = float((pvi[-1] / ema(pvi, 26)[-1] - 1) * 100)
    alarma = azul < 0 and verde > 0
    return {"azul": azul, "verde": verde, "alarma_distribucion": alarma,
            "pasa_capa1": not alarma}


# ============================================================
# 7. RIESGO: ATR% y BETA
# ============================================================

def calcular_atr_pct(velas_diarias: list, periodos: int = 14) -> float | None:
    """ATR(14) como % del precio — para dimensionar stop y posición."""
    v = limpiar_velas(velas_diarias)
    if len(v) < periodos + 1:
        return None
    c = cierres(v)
    h = np.array([x["h"] for x in v])
    l = np.array([x["l"] for x in v])
    tr = np.maximum(h[1:] - l[1:],
                     np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    return float(np.mean(tr[-periodos:]) / c[-1] * 100)


def calcular_beta(velas_semanales: list, velas_benchmark_sem: list, semanas: int = 104) -> float | None:
    """Beta semanal frente al benchmark (STOXX50E en Europa, SPY en EEUU)."""
    w = limpiar_velas(velas_semanales)
    b = limpiar_velas(velas_benchmark_sem)
    if len(w) < 20 or len(b) < 20:
        return None
    cw, cb = cierres(w), cierres(b)
    rw = np.diff(cw) / cw[:-1]
    rb = np.diff(cb) / cb[:-1]
    n = min(len(rw), len(rb), semanas)
    return float(np.cov(rw[-n:], rb[-n:])[0, 1] / np.var(rb[-n:]))


def calcular_liquidez_diaria(velas_diarias: list) -> float | None:
    """Volumen medio en $/€ de 21 sesiones. Umbral: $20M EEUU, €5M Europa
    (parámetros no validados, Sección 0.4)."""
    v = limpiar_velas(velas_diarias)
    if len(v) < 21:
        return None
    c = cierres(v)
    vol = np.array([x.get("v", 0) or 0 for x in v])
    return float(np.mean(vol[-21:] * c[-21:]))


# ============================================================
# 8. CORRELACIÓN ENTRE FINALISTAS
# ============================================================

def matriz_correlacion(series_semanales: dict[str, list], semanas: int = 52) -> dict:
    """series_semanales: {ticker: lista_de_velas_semanales_ya_limpias}
    Devuelve matriz ticker x ticker de correlación de retornos."""
    retornos = {}
    for tk, velas in series_semanales.items():
        c = cierres(limpiar_velas(velas))
        retornos[tk] = np.diff(c) / c[:-1]
    tickers = list(retornos.keys())
    matriz = {}
    for a in tickers:
        matriz[a] = {}
        for b in tickers:
            n = min(len(retornos[a]), len(retornos[b]), semanas)
            matriz[a][b] = float(np.corrcoef(retornos[a][-n:], retornos[b][-n:])[0, 1])
    return matriz


# ============================================================
# 9. ORQUESTADOR — evalúa UN candidato con el checklist completo
# ============================================================

def evaluar_candidato(velas_diarias: list, velas_semanales: list,
                       velas_benchmark_sem: list, precio_max: float = 200.0,
                       liquidez_min: float = 20e6, dias_evento: int | None = None) -> dict:
    """Ejecuta el bloque puente completo (Paso 2 de la Sección 0.4) sobre
    un candidato y devuelve el checklist con las 7 filas + veredicto.
    NO incluye insiders/institucional/valoración — eso sigue siendo
    bloque web, fuera de esta función.
    """
    resultado = {"checklist_incompleto": False, "descartado_en": None}

    sc = calcular_score(velas_diarias)
    if sc is None:
        resultado["descartado_en"] = "sin datos suficientes (precio)"
        return resultado
    resultado.update(sc)

    if sc["precio"] >= precio_max:
        resultado["descartado_en"] = f"precio >= {precio_max}"
        return resultado

    liq = calcular_liquidez_diaria(velas_diarias)
    resultado["liquidez_dia"] = liq
    if liq is not None and liq < liquidez_min:
        resultado["descartado_en"] = f"liquidez < {liquidez_min/1e6:.0f}M/dia"
        return resultado

    if dias_evento is not None and 0 <= dias_evento < 5:
        resultado["descartado_en"] = f"resultados en {dias_evento} dias"
        return resultado
    resultado["dias_evento"] = dias_evento

    dual = calcular_momentum_dual(velas_semanales, velas_benchmark_sem)
    if dual is None or not dual["pasa"]:
        resultado["descartado_en"] = "momentum absoluto <= 0"
        resultado["momentum_dual"] = dual
        return resultado
    resultado["momentum_dual"] = dual

    wein = calcular_etapa_weinstein(velas_semanales)
    if wein is None or not wein["pasa_capa1"]:
        resultado["descartado_en"] = f"Weinstein: {wein['etapa'] if wein else 'sin datos'}"
        resultado["weinstein"] = wein
        return resultado
    resultado["weinstein"] = wein

    konc = calcular_koncorde_proxy(velas_semanales)
    if konc is None or not konc["pasa_capa1"]:
        resultado["descartado_en"] = "Koncorde: alarma de distribución"
        resultado["koncorde"] = konc
        return resultado
    resultado["koncorde"] = konc

    # A partir de aquí, el candidato SOBREVIVE el bloque puente
    resultado["id_fip"] = calcular_fip(velas_diarias)
    resultado["fresco"] = es_momentum_fresco(velas_semanales)
    resultado["atr_pct"] = calcular_atr_pct(velas_diarias)
    resultado["beta"] = calcular_beta(velas_semanales, velas_benchmark_sem)
    resultado["sobrevive_bloque_puente"] = True
    resultado["descartado_en"] = None
    return resultado


if __name__ == "__main__":
    print(__doc__)
    print("Funciones disponibles:")
    for nombre in sorted(dir()):
        if not nombre.startswith("_") and callable(eval(nombre)):
            print(f"  - {nombre}")
