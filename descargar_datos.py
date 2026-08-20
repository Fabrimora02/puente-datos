"""
PUENTE DE DATOS — descargar_datos.py
=====================================
Qué hace: lee la lista de activos de tickers.txt, descarga sus velas
(15 minutos, 1 hora y diario) usando yfinance, y guarda cada serie
como un archivo JSON auto-fechado en la carpeta datos/.

Se ejecuta automáticamente vía GitHub Actions (ver actualizar-datos.yml),
pero también puedes lanzarlo a mano en tu ordenador con:  python descargar_datos.py
"""

# ---------- IMPORTS: las herramientas que usamos ----------
# json: para convertir datos de Python a formato JSON (texto estructurado)
import json
# datetime: para poner el sello de tiempo de "cuándo se generó este archivo"
from datetime import datetime, timezone
# pathlib.Path: forma moderna de manejar rutas de archivos (mejor que strings sueltos)
from pathlib import Path
# yfinance: la librería que descarga datos de mercado de Yahoo Finance.
# La comunidad la mantiene y cubre casi cualquier bolsa del mundo:
# "SM" (NYSE), "ITX.MC" (Madrid), "SIE.DE" (Xetra), "BTC-USD" (cripto), "^VIX" (índices)...
import yfinance as yf

# ---------- CONFIGURACIÓN ----------
# Cada intervalo se descarga con un periodo distinto. ¿Por qué?
# Yahoo limita el intradía: las velas de 15m solo existen para los últimos ~60 días.
# Además no necesitamos archivos gigantes: para operar basta el tramo reciente.
INTERVALOS = {
    "15m": "5d",
    "1h":  "1mo",
    "1d":  "6mo",
    "1wk": "5y",    # NUEVO: semanal, 5 años — el tablero
    "1mo": "max",   # NUEVO: mensual, todo el histórico — el superciclo
}

# Path(__file__).parent = "la carpeta donde vive este script".
# Así el script funciona igual en tu PC y en el servidor de GitHub.
CARPETA_BASE = Path(__file__).parent
CARPETA_DATOS = CARPETA_BASE / "datos"      # aquí se guardan los JSON
ARCHIVO_TICKERS = CARPETA_BASE / "tickers.txt"


def leer_tickers() -> list[str]:
    """Lee tickers.txt y devuelve una lista limpia de tickers.

    Ignora líneas vacías y líneas que empiezan por '#' (comentarios),
    para que puedas anotar cosas en el archivo sin romper nada.
    """
    lineas = ARCHIVO_TICKERS.read_text(encoding="utf-8").splitlines()
    tickers = []
    for linea in lineas:
        limpia = linea.strip()              # quita espacios a los lados
        if limpia and not limpia.startswith("#"):
            tickers.append(limpia)
    return tickers


def descargar_un_activo(ticker: str, intervalo: str, periodo: str) -> dict | None:
    """Descarga las velas de UN activo en UN intervalo y las devuelve como diccionario.

    Devuelve None si algo falla (ticker mal escrito, red caída...) para que
    un fallo en un activo no tumbe la descarga de todos los demás.
    """
    try:
        # yf.Ticker(...) crea el objeto del activo; .history() descarga las velas
        # auto_adjust=False → precios tal cual cotizaron (sin ajustar por dividendos),
        # que es lo que quieres para contar ondas sobre el gráfico real.
        velas = yf.Ticker(ticker).history(
            period=periodo, interval=intervalo, auto_adjust=False
        )

        if velas.empty:  # DataFrame vacío = Yahoo no devolvió nada
            print(f"  AVISO: {ticker} [{intervalo}] llegó vacío — ¿ticker correcto?")
            return None

        # Convertimos el DataFrame de pandas a una lista de velas "planas".
        # iterrows() recorre fila a fila: 'indice' es la fecha, 'fila' los valores OHLCV.
        lista_velas = []
        for indice, fila in velas.iterrows():
            lista_velas.append({
                # isoformat() → "2026-07-13T15:30:00+02:00": legible y sin ambigüedad
                "t": indice.isoformat(),
                "o": round(float(fila["Open"]), 4),
                "h": round(float(fila["High"]), 4),
                "l": round(float(fila["Low"]), 4),
                "c": round(float(fila["Close"]), 4),
                "v": int(fila["Volume"]),
            })

        # El TRIPLE SELLO del protocolo, incorporado de fábrica:
        # generado_utc dice cuándo se creó el archivo → la frescura es siempre medible.
        return {
            "ticker": ticker,
            "intervalo": intervalo,
            "generado_utc": datetime.now(timezone.utc).isoformat(),
            "n_velas": len(lista_velas),
            "primera_vela": lista_velas[0]["t"],
            "ultima_vela": lista_velas[-1]["t"],
            "velas": lista_velas,
        }

    except Exception as error:  # cualquier fallo: lo anotamos y seguimos
        print(f"  ERROR con {ticker} [{intervalo}]: {error}")
        return None

def generar_calendario(tickers: list[str]) -> None:
    """Genera _CALENDARIO.json con la próxima fecha de resultados de cada activo.

    Se ejecuta UNA VEZ AL DÍA: si el archivo ya se generó hoy, no hace nada.
    Motivo: las fechas de resultados no cambian cada 30 minutos, y pedir
    445 calendarios en cada ciclo multiplicaría por 34 las llamadas a Yahoo.
    """
    ruta = CARPETA_DATOS / "_CALENDARIO.json"
    hoy = datetime.now(timezone.utc).date().isoformat()

    # ¿Ya lo generamos hoy? Entonces salimos sin gastar llamadas.
    if ruta.exists():
        try:
            previo = json.loads(ruta.read_text(encoding="utf-8"))
            if previo.get("generado_utc", "")[:10] == hoy:
                print("Calendario: ya generado hoy, se omite.")
                return
        except Exception:
            pass  # si el archivo está corrupto, lo regeneramos

    print(f"Calendario: consultando fechas de resultados de {len(tickers)} activos...")
    eventos = {}
    for ticker in tickers:
        try:
            cal = yf.Ticker(ticker).calendar
            fecha = None
            # yfinance devuelve dict o DataFrame según versión y activo
            if isinstance(cal, dict):
                lista = cal.get("Earnings Date") or []
                if lista:
                    fecha = lista[0]
            elif cal is not None and hasattr(cal, "empty") and not cal.empty:
                if "Earnings Date" in cal.index:
                    fecha = cal.loc["Earnings Date"][0]

            if fecha is None:
                continue  # índices, materias primas y divisas no tienen resultados

            fecha_txt = fecha.isoformat()[:10]
            dias = (datetime.fromisoformat(fecha_txt).date()
                    - datetime.now(timezone.utc).date()).days
            eventos[ticker] = {"proxima_fecha": fecha_txt, "dias_restantes": dias}

        except Exception:
            continue  # un fallo individual nunca rompe la ejecución

    ruta.write_text(
        json.dumps({
            "generado_utc": datetime.now(timezone.utc).isoformat(),
            "n_activos_con_fecha": len(eventos),
            "eventos": eventos,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Calendario: {len(eventos)} activos con fecha de resultados.")

def main():
    """Función principal: recorre todos los tickers y todos los intervalos."""
    CARPETA_DATOS.mkdir(exist_ok=True)  # crea la carpeta datos/ si no existe
    tickers = leer_tickers()
    print(f"Descargando {len(tickers)} activos: {', '.join(tickers)}")

    exitos, fallos = 0, 0
    indice = []  # aquí acumulamos un resumen de cada archivo para el manifiesto final
    for ticker in tickers:
        for intervalo, periodo in INTERVALOS.items():
            datos = descargar_un_activo(ticker, intervalo, periodo)
            if datos is None:
                fallos += 1
                continue

            # Los puntos y símbolos raros del ticker se cambian por '-' para
            # que el nombre de archivo sea limpio: "ITX.MC" → "ITX-MC_15m.json"
            nombre = ticker.replace(".", "-").replace("^", "").replace("=", "-")
            ruta = CARPETA_DATOS / f"{nombre}_{intervalo}.json"

            # indent=2 → JSON legible por humanos; ensure_ascii=False → acentos OK
            ruta.write_text(
                json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"  OK: {ruta.name} ({datos['n_velas']} velas, "
                  f"última: {datos['ultima_vela']})")
            exitos += 1

            # Anotamos la ficha de este archivo en el índice general
            indice.append({
                "archivo": ruta.name,
                "ticker": ticker,
                "intervalo": intervalo,
                "ultima_vela": datos["ultima_vela"],
                "n_velas": datos["n_velas"],
            })

    # MANIFIESTO: un único archivo (_INDICE.json) con la foto completa del puente.
    # Leyendo solo este archivo, Claude sabe qué activos hay, con qué intervalos,
    # y la frescura de cada uno — sin tener que abrir los JSON uno a uno.
    manifiesto = {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "total_archivos": exitos,
        "fallos": fallos,
        "archivos": indice,
    }
    (CARPETA_DATOS / "_INDICE.json").write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    generar_calendario(tickers)
    print(f"\nResumen: {exitos} archivos escritos, {fallos} fallos. Índice actualizado.")


# Este 'if' es un clásico de Python: el bloque solo se ejecuta si lanzas
# el script directamente (python descargar_datos.py), no si otro código lo importa.
if __name__ == "__main__":
    main()
