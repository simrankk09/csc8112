import os, json, time, queue, threading
import requests
import paho.mqtt.client as mqtt

HOST = os.getenv("MQTT_HOST", "localhost")
URL = os.getenv("DASHBOARD_URL", "http://dashboard.cloud.ncl.ac.uk:8000/api/records")
THRESHOLD = int(os.getenv("THRESHOLD", "5"))

latest = {"image": None, "reading": None}
post_q = queue.Queue()

def on_connect(c, u, f, rc):
    c.subscribe([("CSC8112/annotated", 0), ("CSC8112/vehicle_count", 0), ("CSC8112/quality", 0)])
    print("decision module connected", flush=True)

def on_message(c, u, m):
    if m.topic == "CSC8112/annotated":
        latest["image"] = bytes(m.payload)
    elif m.topic == "CSC8112/quality":
        try:
            latest["reading"] = json.loads(m.payload)
        except Exception:
            pass
    elif m.topic == "CSC8112/vehicle_count":
        try:
            n = int(json.loads(m.payload)["vehicles_count"])
        except Exception:
            return
        if n <= THRESHOLD:
            print("vehicles", n, "<=", THRESHOLD, "-> frame dropped", flush=True)
            return
        if latest["image"] is None or latest["reading"] is None:
            print("vehicles", n, "but image or reading missing -> frame skipped", flush=True)
            return
        post_q.put((latest["image"], dict(latest["reading"]), n))
        print("vehicles", n, "> threshold -> queued for dashboard", flush=True)

def sender():
    while True:
        img, rd, n = post_q.get()
        data = {
            "timestamp": int(float(rd["Timestamp"]) / 1000),
            "value": rd["Value"],
            "predicted_quality": rd["Quality"],
        }
        try:
            r = requests.post(URL, files={"image": ("frame.jpg", img, "image/jpeg")}, data=data, timeout=20)
            print("POST", data, "->", r.status_code, r.text[:300], flush=True)
        except Exception as e:
            print("POST failed:", e, flush=True)

threading.Thread(target=sender, daemon=True).start()

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(HOST, 1883)
client.loop_forever()
