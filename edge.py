import os, json, time
import numpy as np
import paho.mqtt.client as mqtt
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ai_edge_litert.interpreter import Interpreter

HOST = os.getenv("MQTT_HOST", "localhost")
OUT = os.getenv("OUT_DIR", "/output")
IDLE = int(os.getenv("IDLE_SECONDS", "90"))
CLASSES = ["GREEN", "RED", "YELLOW"]
COLORS = {"GREEN": "green", "YELLOW": "gold", "RED": "red"}
os.makedirs(OUT, exist_ok=True)

interp = Interpreter(model_path="model.tflite")
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

def classify(values):
    x = (np.array(values, dtype=np.float32) / 100.0).reshape(-1, 1)
    interp.resize_tensor_input(inp["index"], [len(x), 1])
    interp.allocate_tensors()
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    p = interp.get_tensor(out["index"])
    return [CLASSES[i] for i in p.argmax(axis=1)]

buf, seen, rows = [], set(), []
last = [time.time()]

def on_connect(c, u, f, rc):
    c.subscribe("CSC8112")

def on_message(c, u, m):
    try:
        d = json.loads(m.payload)
        ts, v = d["Timestamp"], float(d["Value"])
    except Exception:
        return
    if ts in seen:
        return
    seen.add(ts)
    buf.append((ts, v))
    last[0] = time.time()

def make_charts():
    df = pd.DataFrame(rows, columns=["Timestamp", "Value", "Quality"]).sort_values("Timestamp")
    df["Time"] = pd.to_datetime(df["Timestamp"], unit="ms")
    plt.figure(figsize=(12, 5))
    plt.plot(df["Time"], df["Value"])
    plt.title("PM2.5 over time"); plt.xlabel("Time"); plt.ylabel("PM2.5")
    plt.savefig(os.path.join(OUT, "chart1_line.png")); plt.close()
    counts = df["Quality"].value_counts()
    cols = [COLORS[c] for c in counts.index]
    plt.figure()
    plt.bar(counts.index, counts.values, color=cols)
    plt.title("Readings per air quality class"); plt.ylabel("Count")
    plt.savefig(os.path.join(OUT, "chart2_bar.png")); plt.close()
    plt.figure()
    plt.pie(counts.values, labels=counts.index, colors=cols, autopct="%1.1f%%")
    plt.title("Share of each air quality class")
    plt.savefig(os.path.join(OUT, "chart3_pie.png")); plt.close()
    plt.figure(figsize=(12, 5))
    for q, g in df.groupby("Quality"):
        plt.scatter(g["Time"], g["Value"], s=6, c=COLORS[q], label=q)
    plt.legend(); plt.title("PM2.5 coloured by class"); plt.xlabel("Time"); plt.ylabel("PM2.5")
    plt.savefig(os.path.join(OUT, "chart4_scatter.png")); plt.close()
    print("charts saved to", OUT, flush=True)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(HOST, 1883)
client.loop_start()

charts_done = False
while True:
    time.sleep(3)
    batch = buf[:]
    del buf[:len(batch)]
    if batch:
        labels = classify([v for _, v in batch])
        for (ts, v), q in zip(batch, labels):
            rows.append((ts, v, q))
            client.publish("CSC8112/quality", json.dumps({"Timestamp": ts, "Value": v, "Quality": q}))
        summary = {c: labels.count(c) for c in CLASSES}
        print("batch", len(batch), summary, "| last:", batch[-1][1], labels[-1], "| total:", len(rows), flush=True)
    elif rows and not charts_done and time.time() - last[0] > IDLE:
        make_charts()
        charts_done = True
