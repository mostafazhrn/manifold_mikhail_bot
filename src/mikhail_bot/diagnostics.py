# diagnostics.py
import pickle, json, numpy as np, os
MODEL_PATH = os.path.join("models", "model.pkl")
V_PATH = os.path.join("models", "vectorizer.pkl")
S_PATH = os.path.join("models", "scaler.pkl")

with open(MODEL_PATH,"rb") as f: model = pickle.load(f)
with open(V_PATH,"rb") as f: vect = pickle.load(f)
with open(S_PATH,"rb") as f: scaler = pickle.load(f)

# load resolved markets (a small sample)
samples = []
with open("data/resolved_markets.jsonl","r",encoding="utf8") as f:
    for i,line in enumerate(f):
        if i>2000: break
        try:
            samples.append(json.loads(line))
        except: pass

probs=[]
market_ps=[]
for m in samples:
    q = m.get("question","")
    X_text = vect.transform([q]).toarray()
    X_num = scaler.transform([[
        float(m.get("probability",0)),
        float(m.get("p",0)),
        float(m.get("pool",{}).get("YES",0)),
        float(m.get("pool",{}).get("NO",0)),
        float(m.get("volume",0)),
        float(m.get("volume24Hours",0)),
        float(m.get("totalLiquidity",0)),
        float(m.get("uniqueBettorCount",0)),
        float(m.get("closeTime",0)),
        float(m.get("createdTime",0)),
    ]])
    X = np.hstack([X_text, X_num])
    try:
        prob = model.predict_proba(X)[0][1]
    except Exception as e:
        continue
    probs.append(prob)
    market_ps.append(float(m.get("probability") or m.get("p") or 0.5))

probs = np.array(probs)
market_ps = np.array(market_ps)
print("samples:", len(probs))
print("model prob mean/std:", probs.mean(), probs.std())
print("market p mean/std:", market_ps.mean(), market_ps.std())
print("fraction model prob < 0.5:", (probs < 0.5).mean())
print("fraction market p < 0.5:", (market_ps < 0.5).mean())
# optionally save arrays for plotting offline
np.save("logs/model_probs.npy", probs)
np.save("logs/market_ps.npy", market_ps)
