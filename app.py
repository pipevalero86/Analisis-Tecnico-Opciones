import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st

# Cargar datos históricos desde Yahoo Finance con caché
@st.cache_data
def obtener_datos_historicos(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period="6mo")

# Obtener fechas de expiración de las opciones (retorna solo las fechas serializables)
@st.cache_data
def obtener_opciones(ticker):
    stock = yf.Ticker(ticker)
    expiraciones = stock.options
    return expiraciones

# GRIEGOS con manejo de datos faltantes mejorado
def obtener_griegos(ticker, fecha_expiracion, tipo_opcion='CALL'):
    stock = yf.Ticker(ticker)
    opciones = stock.option_chain(date=fecha_expiracion)
    
    # Comprobamos si las columnas de griegos existen
    columnas_griegos = ['strike', 'lastPrice', 'delta', 'gamma', 'theta', 'vega', 'rho']

    if tipo_opcion == 'CALL':
        data = opciones.calls
    else:
        data = opciones.puts
    
    # Filtrar solo las columnas disponibles en el DataFrame
    columnas_disponibles = [col for col in columnas_griegos if col in data.columns]
    
    # Verificar si algunas columnas de los griegos faltan y mostrar un aviso
    columnas_faltantes = set(columnas_griegos) - set(columnas_disponibles)
    if columnas_faltantes:
        st.warning(f"Algunas columnas de griegos no están disponibles: {', '.join(columnas_faltantes)}")

    # Verificar si las columnas disponibles tienen datos no nulos
    if columnas_disponibles:
        data = data[columnas_disponibles].dropna(how='all')  # Eliminar filas con todos los valores nulos
        if data.empty:
            st.warning("No hay datos de griegos disponibles para este strike y fecha de vencimiento.")
            return pd.DataFrame()  # Retornar DataFrame vacío si no hay datos
        return data.round(2)  # Redondear valores si están disponibles
    else:
        st.warning("No se encontraron datos de griegos para las opciones seleccionadas.")
        return pd.DataFrame()  # Retornar DataFrame vacío si no hay columnas disponibles

# LIQUIDEZ (volumen, interés abierto, spread bid/ask)
def obtener_liquidez(ticker, fecha_expiracion, tipo_opcion='CALL'):
    stock = yf.Ticker(ticker)
    opciones = stock.option_chain(date=fecha_expiracion)

    if tipo_opcion == 'CALL':
        data = opciones.calls[['strike', 'volume', 'openInterest', 'bid', 'ask']]
    else:
        data = opciones.puts[['strike', 'volume', 'openInterest', 'bid', 'ask']]

    # Calcular el spread bid/ask
    data['spread'] = (data['ask'] - data['bid']).round(2)

    return data

# GRUPO: TÉCNICOS - Medias Móviles (SMA y EMA)
def calcular_sma(data, period=50):
    return data['Close'].rolling(window=period).mean()

def calcular_ema(data, period=20):
    return data['Close'].ewm(span=period, adjust=False).mean()

# GRUPO: TÉCNICOS - RSI
def calcular_rsi(data, period=14):
    delta = data['Close'].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# GRUPO: TÉCNICOS - Bandas de Bollinger
def calcular_bollinger_bands(data, period=20, std_dev=2):
    sma = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    return upper_band, lower_band

# GRUPO: TÉCNICOS - MACD
def calcular_macd(data, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = data['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line, signal_line

# GRUPO: TÉCNICOS - ATR
def calcular_atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

# Otros Indicadores - Put/Call Ratio
def calcular_put_call_ratio(ticker, fecha_expiracion):
    stock = yf.Ticker(ticker)
    opciones = stock.option_chain(date=fecha_expiracion)
    
    vol_calls = opciones.calls['volume'].sum()
    vol_puts = opciones.puts['volume'].sum()

    ratio = vol_puts / vol_calls if vol_calls != 0 else 0
    return round(ratio, 2)

# Función para calcular la volatilidad implícita
def calcular_volatilidad_implicita(ticker, fecha_expiracion, tipo_opcion):
    stock = yf.Ticker(ticker)
    opciones = stock.option_chain(date=fecha_expiracion)

    if tipo_opcion == 'CALL':
        volatilidad = opciones.calls['impliedVolatility']
    else:
        volatilidad = opciones.puts['impliedVolatility']

    return round(volatilidad.mean(), 2)

# Otros Indicadores - Skew de Volatilidad
def calcular_skew_volatilidad(ticker, fecha_expiracion):
    stock = yf.Ticker(ticker)
    opciones = stock.option_chain(date=fecha_expiracion)
    
    iv_calls = opciones.calls['impliedVolatility'].mean()
    iv_puts = opciones.puts['impliedVolatility'].mean()

    skew = (iv_puts - iv_calls) * 100
    return round(skew, 2)

# Otros Indicadores - Acumulación/Distribución (AD)
def calcular_acumulacion_distribucion(data):
    ad = (data['Close'] - data['Low']) - (data['High'] - data['Close'])
    ad /= (data['High'] - data['Low'])
    ad = ad * data['Volume']
    return ad.cumsum().iloc[-1].round(2)

# Interpretaciones mejoradas de los indicadores
def interpretar_rangos(valor, rango_min, rango_max, nombre_indicador):
    if valor < rango_min:
        if nombre_indicador == "RSI":
            return f"{valor:.2f} está por debajo del rango óptimo [{rango_min}, {rango_max}]. Esto indica sobreventa, lo que puede ser una oportunidad para comprar si se espera una reversión alcista."
        elif nombre_indicador == "MACD":
            return f"{valor:.2f} está por debajo del rango óptimo [{rango_min}, {rango_max}]. Esto indica un momentum bajista."
        elif nombre_indicador == "ATR":
            return f"{valor:.2f} está por debajo del rango óptimo [{rango_min}, {rango_max}]. Esto indica baja volatilidad, lo que sugiere un mercado calmado."
        elif nombre_indicador == "Put/Call Ratio":
            return f"{valor:.2f} está por debajo del rango óptimo [{rango_min}, {rango_max}]. Esto indica un sentimiento alcista, con más interés en opciones CALL."
        elif nombre_indicador == "Volatilidad Implícita":
            return f"{valor:.2f} está por debajo del rango óptimo [{rango_min}, {rango_max}]. Esto sugiere una baja volatilidad implícita, lo que puede significar que el mercado no espera grandes movimientos."
        elif nombre_indicador == "Skew de Volatilidad":
            return f"{valor:.2f} está por debajo del rango óptimo [{rango_min}, {rango_max}]. Esto indica un sesgo bajista, ya que las PUTs tienen mayor volatilidad implícita."
    elif valor > rango_max:
        if nombre_indicador == "RSI":
            return f"{valor:.2f} está por encima del rango óptimo [{rango_min}, {rango_max}]. Esto indica sobrecompra, lo que puede señalar una posible reversión bajista."
        elif nombre_indicador == "MACD":
            return f"{valor:.2f} está por encima del rango óptimo [{rango_min}, {rango_max}]. Esto sugiere un fuerte momentum alcista."
        elif nombre_indicador == "ATR":
            return f"{valor:.2f} está por encima del rango óptimo [{rango_min}, {rango_max}]. Esto indica alta volatilidad, lo que sugiere un mercado más turbulento."
        elif nombre_indicador == "Put/Call Ratio":
            return f"{valor:.2f} está por encima del rango óptimo [{rango_min}, {rango_max}]. Esto indica un sentimiento bajista, con más interés en opciones PUT."
        elif nombre_indicador == "Volatilidad Implícita":
            return f"{valor:.2f} está por encima del rango óptimo [{rango_min}, {rango_max}]. Esto sugiere una alta volatilidad implícita, lo que puede indicar que el mercado espera grandes movimientos."
        elif nombre_indicador == "Skew de Volatilidad":
            return f"{valor:.2f} está por encima del rango óptimo [{rango_min}, {rango_max}]. Esto indica un sesgo alcista, ya que las CALLs tienen mayor volatilidad implícita."
    else:
        if nombre_indicador == "RSI":
            return f"{valor:.2f} está dentro del rango óptimo [{rango_min}, {rango_max}]. El RSI está en un rango neutral, sin señales claras de sobrecompra o sobreventa."
        elif nombre_indicador == "MACD":
            return f"{valor:.2f} está dentro del rango óptimo [{rango_min}, {rango_max}]. El MACD indica un momentum estable."
        elif nombre_indicador == "ATR":
            return f"{valor:.2f} está dentro del rango óptimo [{rango_min}, {rango_max}]. Esto sugiere una volatilidad moderada en el mercado."
        elif nombre_indicador == "Put/Call Ratio":
            return f"{valor:.2f} está dentro del rango óptimo [{rango_min}, {rango_max}]. El sentimiento del mercado es neutral entre opciones CALL y PUT."
        elif nombre_indicador == "Volatilidad Implícita":
            return f"{valor:.2f} está dentro del rango óptimo [{rango_min}, {rango_max}]. La volatilidad implícita está en niveles normales, sin grandes expectativas de cambios drásticos."
        elif nombre_indicador == "Skew de Volatilidad":
            return f"{valor:.2f} está dentro del rango óptimo [{rango_min}, {rango_max}]. No hay un sesgo claro en la volatilidad implícita entre opciones CALL y PUT."

# Función para mostrar recomendaciones y nombres de los indicadores
def mostrar_resultado(nombre, valor, recomendacion):
    st.write(f"**{nombre}**: {valor:.2f} - {recomendacion}")

# Interpretación para SMA y EMA mejorada
def interpretar_sma_ema(precio_actual, sma, ema):
    interpretacion = ""
    
    if precio_actual > sma:
        interpretacion += "El precio está por encima de la SMA, indicando tendencia alcista.\n"
    else:
        interpretacion += "El precio está por debajo de la SMA, indicando tendencia bajista.\n"
    
    if ema > sma:
        interpretacion += "La EMA está por encima de la SMA, confirmando una tendencia alcista de corto plazo."
    else:
        interpretacion += "La EMA está por debajo de la SMA, sugiriendo una tendencia bajista de corto plazo."
    
    return interpretacion

# Interpretación para Bandas de Bollinger
def interpretar_bollinger(precio_actual, upper_band, lower_band):
    if precio_actual >= upper_band:
        return "El precio está tocando la banda superior, indicando sobrecompra."
    elif precio_actual <= lower_band:
        return "El precio está tocando la banda inferior, indicando sobreventa."
    else:
        return "El precio está dentro de las bandas, indicando una volatilidad moderada."

# Mostrar los indicadores faltantes (SMA, EMA, y Bandas de Bollinger)
def mostrar_indicadores_faltantes(precio_actual, sma, ema, upper_band, lower_band):
    st.subheader("Medias Móviles (SMA y EMA)")
    mostrar_resultado("SMA", sma, interpretar_sma_ema(precio_actual, sma, ema))
    mostrar_resultado("EMA", ema, interpretar_sma_ema(precio_actual, sma, ema))
    
    st.subheader("Bandas de Bollinger")
    mostrar_resultado("Banda Superior", upper_band, interpretar_bollinger(precio_actual, upper_band, lower_band))
    mostrar_resultado("Banda Inferior", lower_band, interpretar_bollinger(precio_actual, upper_band, lower_band))

# Interpretaciones mejoradas para los indicadores de liquidez
def interpretar_liquidez(volumen, open_interest, spread):
    interpretaciones = {}
    
    if volumen > 1000:
        interpretaciones['volumen'] = f"{volumen} - El volumen es alto, lo que sugiere que esta opción tiene buena liquidez."
    else:
        interpretaciones['volumen'] = f"{volumen} - El volumen es bajo, lo que puede significar menos interés y menor liquidez en esta opción."

    if open_interest > 500:
        interpretaciones['open_interest'] = f"{open_interest} - El interés abierto es alto, lo que sugiere una fuerte participación en esta opción."
    else:
        interpretaciones['open_interest'] = f"{open_interest} - El interés abierto es bajo, lo que puede indicar menos interés y menor liquidez en esta opción."

    if spread < 0.5:
        interpretaciones['spread'] = f"{spread:.2f} - El spread es bajo, lo que indica buena liquidez y precios de compra/venta cercanos."
    else:
        interpretaciones['spread'] = f"{spread:.2f} - El spread es amplio, lo que puede dificultar la ejecución de órdenes a un precio favorable."
    
    return interpretaciones

# Inicializar la lista de tickers usados en session_state
if 'tickers_usados' not in st.session_state:
    st.session_state['tickers_usados'] = []

# Función principal para mostrar todos los indicadores
def main():
    st.title("Análisis Completo de Opciones (Griegos, Liquidez, Técnicos y Otros)")

    # Inicializar la lista de tickers usados en session_state si no existe
    if 'tickers_usados' not in st.session_state:
        st.session_state['tickers_usados'] = []

    # Crear una opción para usar un ticker previo o ingresar uno nuevo
    usar_ticker_previo = st.checkbox("¿Usar un ticker previo?", False)

    if usar_ticker_previo and st.session_state['tickers_usados']:
        # Si el usuario quiere usar un ticker previo y hay tickers en la lista
        ticker_seleccionado = st.selectbox("Selecciona un ticker previo:", st.session_state['tickers_usados'])
    else:
        # Si el usuario quiere ingresar un nuevo ticker
        ticker_seleccionado = st.text_input("Ingresa el ticker:", "SPY")

    # Agregar el nuevo ticker a la lista si no está ya
    if ticker_seleccionado and ticker_seleccionado not in st.session_state['tickers_usados']:
        st.session_state['tickers_usados'].append(ticker_seleccionado)

    expiraciones = obtener_opciones(ticker_seleccionado)

    if ticker_seleccionado and expiraciones:
        fecha_expiracion = st.selectbox("Selecciona la fecha de vencimiento:", expiraciones)
        stock = yf.Ticker(ticker_seleccionado)
        tipo_opcion = st.selectbox("Selecciona el tipo de opción:", ["CALL", "PUT"])

        # Obtener el precio actual de la acción
        precio_actual = stock.history(period="1d")['Close'].iloc[-1]

        # Corrección: Seleccionar strikes según tipo de opción
        if tipo_opcion == 'CALL':
            strikes = stock.option_chain(fecha_expiracion).calls['strike']
        else:
            strikes = stock.option_chain(fecha_expiracion).puts['strike']
        
        # Encontrar el strike más cercano al precio actual (ATM)
        strike_atm = strikes.iloc[(strikes - precio_actual).abs().argsort()[:1]].values[0]

        # Seleccionar el strike ATM por defecto
        strike_price = st.selectbox("Selecciona el strike price:", strikes, index=strikes.tolist().index(strike_atm))

        if fecha_expiracion and tipo_opcion and strike_price:
            datos_historicos = obtener_datos_historicos(ticker_seleccionado)

            ### Sección 1: Griegos ###
            st.subheader("Griegos")
            griegos = obtener_griegos(ticker_seleccionado, fecha_expiracion, tipo_opcion)
            griegos = griegos[griegos['strike'] == strike_price]  # Filtrar por strike seleccionado
            if not griegos.empty:
                st.write(griegos)
            else:
                st.write("No se encontraron datos de griegos para las opciones seleccionadas.")

            ### Sección 2: Indicadores de Liquidez ###
            st.subheader("Indicadores de Liquidez")
            liquidez = obtener_liquidez(ticker_seleccionado, fecha_expiracion, tipo_opcion)
            liquidez = liquidez[liquidez['strike'] == strike_price]  # Filtrar por strike seleccionado
            st.write(liquidez)
            
            # Interpretar indicadores de liquidez
            interpretaciones_liquidez = interpretar_liquidez(
                liquidez['volume'].iloc[0],  # Volumen
                liquidez['openInterest'].iloc[0],  # Interés abierto
                liquidez['spread'].iloc[0]  # Spread bid/ask
            )

            # Mostrar interpretación de los indicadores de liquidez
            st.write(f"Volumen: {interpretaciones_liquidez['volumen']}")
            st.write(f"Interés Abierto: {interpretaciones_liquidez['open_interest']}")
            st.write(f"Spread Bid/Ask: {interpretaciones_liquidez['spread']}")

            ### Sección 3: Indicadores Técnicos ###
            st.subheader("Indicadores Técnicos")
            sma_50 = calcular_sma(datos_historicos)
            ema_20 = calcular_ema(datos_historicos)
            rsi = calcular_rsi(datos_historicos)
            upper_band, lower_band = calcular_bollinger_bands(datos_historicos)
            macd_line, signal_line = calcular_macd(datos_historicos)
            atr = calcular_atr(datos_historicos)

            # Mostrar SMA, EMA y Bandas de Bollinger
            mostrar_indicadores_faltantes(precio_actual, sma_50.iloc[-1], ema_20.iloc[-1], upper_band.iloc[-1], lower_band.iloc[-1])
            mostrar_resultado("RSI", rsi.iloc[-1], interpretar_rangos(rsi.iloc[-1], 30, 70, "RSI"))
            mostrar_resultado("MACD", macd_line.iloc[-1], interpretar_rangos(macd_line.iloc[-1], -1, 1, "MACD"))
            mostrar_resultado("ATR", atr.iloc[-1], interpretar_rangos(atr.iloc[-1], 0, 5, "ATR"))

            ### Sección 4: Otros Indicadores ###
            st.subheader("Otros Indicadores")
            put_call_ratio = calcular_put_call_ratio(ticker_seleccionado, fecha_expiracion)
            iv = calcular_volatilidad_implicita(ticker_seleccionado, fecha_expiracion, tipo_opcion)
            skew = calcular_skew_volatilidad(ticker_seleccionado, fecha_expiracion)
            ad = calcular_acumulacion_distribucion(datos_historicos)

            mostrar_resultado("Put/Call Ratio", put_call_ratio, interpretar_rangos(put_call_ratio, 0.5, 1.5, "Put/Call Ratio"))
            mostrar_resultado("Volatilidad Implícita", iv, interpretar_rangos(iv, 20, 80, "Volatilidad Implícita"))
            mostrar_resultado("Skew de Volatilidad", skew, interpretar_rangos(skew, -10, 10, "Skew de Volatilidad"))
            mostrar_resultado("Acumulación/Distribución", ad, "Indicador de acumulación o distribución")

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
