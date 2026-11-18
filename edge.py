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
    order = ["GREEN", "YELLOW", "RED"]
    counts = df["Quality"].value_counts().reindex(order).fillna(0)

    plt.figure()
    plt.bar(order, counts.values, color=[COLORS[c] for c in order])
    for i, v in enumerate(counts.values):
        plt.text(i, v, int(v), ha="center", va="bottom")
    plt.title("Count of Green / Yellow / Red classifications"); plt.ylabel("Count")
    plt.savefig(os.path.join(OUT, "chart1_bar.png")); plt.close()

    plt.figure(figsize=(12, 5))
    for q in order:
        g = df[df["Quality"] == q]
        plt.scatter(g["Time"], g["Value"], s=6, c=COLORS[q], label=q)
    plt.legend(); plt.title("PM2.5 over time, coloured by classification")
    plt.xlabel("Time"); plt.ylabel("PM2.5")
    plt.savefig(os.path.join(OUT, "chart2_timeseries.png")); plt.close()

    plt.figure()
    plt.hist(df["Value"], bins=40)
    plt.title("PM2.5 value frequency distribution")
    plt.xlabel("PM2.5"); plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUT, "chart3_frequency.png")); plt.close()

    summary = df.groupby("Quality")["Value"].agg(["count", "min", "max", "mean"]).round(2)
    summary["share %"] = (summary["count"] / len(df) * 100).round(1)
    print(summary, flush=True)
    table = summary.reset_index().astype(str).values.tolist()
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis("off")
    tb = ax.table(cellText=table, colLabels=["Class", "Count", "Min", "Max", "Mean", "Share %"], loc="center")
    tb.scale(1, 1.6)
    ax.set_title("Classification summary (total %d readings)" % len(df))
    fig.savefig(os.path.join(OUT, "chart4_summary.png")); plt.close(fig)
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
            print("PM2.5 =", v, "->", q, flush=True)
            client.publish("CSC8112/quality", json.dumps({"Timestamp": ts, "Value": v, "Quality": q}), retain=True)
        summary = {c: labels.count(c) for c in CLASSES}
        print("BATCH of", len(batch), summary, "| total:", len(rows), flush=True)
    elif rows and not charts_done and time.time() - last[0] > IDLE:
        make_charts()
        charts_done = True
