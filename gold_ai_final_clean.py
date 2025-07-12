
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# إعداد الصفحة
st.set_page_config(page_title="تحليل الذهب AI", layout="wide")
st.title("🤖📈 تحليل الذهب (XAU/USD) بالذكاء الاصطناعي")

# تحميل البيانات
data = yf.download("GC=F", interval="15m", period="5d")
if data.empty or 'Close' not in data.columns:
    st.error("⚠️ فشل تحميل البيانات! تأكد من الاتصال أو الرمز.")
    st.stop()

# المؤشرات الفنية
data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
data['MACD'] = ta.trend.MACD(data['Close']).macd()
data['EMA20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
bb = ta.volatility.BollingerBands(data['Close'])
data['Boll_Upper'] = bb.bollinger_hband()
data['Boll_Lower'] = bb.bollinger_lband()
data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
data['Momentum'] = ta.momentum.MomentumIndicator(data['Close']).momentum()
data['Stochastic'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()

# نموذج AI لتوقع الاتجاه
features = ['RSI', 'MACD', 'EMA20', 'Momentum', 'ATR', 'Stochastic']
df_model = data[features].dropna()
X = df_model
y = np.where(data.loc[X.index, 'Close'].shift(-1) > data.loc[X.index, 'Close'], 1, 0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier()
model.fit(X_scaled, y)
latest = X.iloc[-1:]
latest_scaled = scaler.transform(latest)
pred = model.predict(latest_scaled)[0]
direction = "🔼 شراء" if pred == 1 else "🔽 بيع"
close = data['Close'].iloc[-1]
atr = data['ATR'].iloc[-1]
tp = close + atr * 2 if pred == 1 else close - atr * 2
sl = close - atr if pred == 1 else close + atr

# عرض التوصية
st.subheader("📢 التوصية")
st.markdown(f"### الاتجاه المتوقع: {direction}")
st.markdown(f"**TP (الهدف):** {tp:.2f} | **SL (الوقف):** {sl:.2f}")

# تحليل الشموع اليدوي
def detect_doji(o, h, l, c, threshold=0.1):
    return abs(c - o) / (h - l + 1e-6) < threshold

def detect_bullish_engulfing(po, pc, co, cc):
    return pc < po and cc > co and cc > po and co < pc

def detect_hammer(o, h, l, c):
    body = abs(c - o)
    lower = o - l if c > o else c - l
    upper = h - c if c > o else h - o
    return lower > 2 * body and upper < body

def detect_shooting_star(o, h, l, c):
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    return upper > 2 * body and lower < body

last = data.iloc[-2:]
prev = last.iloc[0]
curr = last.iloc[1]

patterns = []
if detect_doji(curr['Open'], curr['High'], curr['Low'], curr['Close']):
    patterns.append("⚠️ Doji")
if detect_bullish_engulfing(prev['Open'], prev['Close'], curr['Open'], curr['Close']):
    patterns.append("🟢 Bullish Engulfing")
if detect_hammer(curr['Open'], curr['High'], curr['Low'], curr['Close']):
    patterns.append("🔨 Hammer")
if detect_shooting_star(curr['Open'], curr['High'], curr['Low'], curr['Close']):
    patterns.append("🌠 Shooting Star")

st.subheader("📌 نماذج الشموع المكتشفة:")
if patterns:
    for p in patterns:
        st.write(p)
else:
    st.write("لا يوجد نمط شمعة واضح")

# رسم الشموع + المؤشرات
fig = go.Figure()
fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                             low=data['Low'], close=data['Close'], name='XAU/USD'))
fig.add_trace(go.Scatter(x=data.index, y=data['EMA20'], mode='lines', name='EMA 20'))
fig.add_trace(go.Scatter(x=data.index, y=data['Boll_Upper'], mode='lines', name='Boll Upper'))
fig.add_trace(go.Scatter(x=data.index, y=data['Boll_Lower'], mode='lines', name='Boll Lower'))
st.plotly_chart(fig, use_container_width=True)

# عرض المؤشرات الأخيرة
st.subheader("📊 بيانات فنية")
st.dataframe(data[['RSI', 'MACD', 'EMA20', 'ATR']].tail(10))
