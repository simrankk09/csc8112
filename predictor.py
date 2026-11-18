import os, json, time, sys
import paho.mqtt.client as mqtt
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ml_engine import MLPredictor

HOST = os.getenv("MQTT_HOST", "localhost")
TARGET = int(os.getenv("TARGET", "8706"))
IDLE = int(os.getenv("IDLE_SECONDS", "180"))
OUT = os.getenv("OUT_DIR", "/output")
os.makedirs(OUT, exist_ok=True)

data = {}
last = [time.time()]

def on_connect(c, u, f, rc):
    c.subscribe("CSC8112")

def on_message(c, u, m):
    try:
        d = json.loads(m.payload)
        ts, v = d["Timestamp"], d["Value"]
    except Exception:
        return
    if ts not in data:
        data[ts] = v
        last[0] = time.time()

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(HOST, 1883)
client.loop_start()

while len(data) < TARGET and time.time() - last[0] < IDLE:
    time.sleep(5)
    print("collected unique points:", len(data), flush=True)
client.loop_stop()

df = pd.DataFrame({"Timestamp": list(data.keys()), "Value": list(data.values())})
df = df.drop_duplicates().sort_values("Timestamp")
print("last 2 values:", df["Value"].tail(2).tolist(), flush=True)

df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
df = df.set_index("Timestamp").resample("10min").mean().dropna().reset_index()
df["Timestamp"] = pd.to_datetime(df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"))
print(df.to_string(), flush=True)

plt.figure(figsize=(12, 5))
plt.plot(df["Timestamp"], df["Value"])
plt.title("PM2.5 averaged over 10 minutes")
plt.xlabel("Time"); plt.ylabel("PM2.5")
plt.savefig(os.path.join(OUT, "pm25_averaged.png"))

predictor = MLPredictor(df)
predictor.train()
forecast = predictor.predict()
print(forecast.tail(12), flush=True)
fig = predictor.plot_result(forecast)
fig.savefig(os.path.join(OUT, "pm25_forecast.png"))
print("done, graphs saved to", OUT, flush=True)
