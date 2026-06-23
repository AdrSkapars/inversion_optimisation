import json, glob, os, collections, statistics
BASE="experiments/bloom/runs_16"
def runs_with_maxturns1():
    found={}
    for cfgp in glob.glob(BASE+"/*/round_*/cfg.json"):
        run=cfgp.split("/round_")[0]
        if run in found: continue
        try:
            c=json.load(open(cfgp)); ro=c.get("rollout") or {}
            mt=ro.get("max_turns"); en=(c.get("corruption_output") or {}).get("enabled")
            found[run]={"max_turns":mt,"corrupt":bool(en)}
        except Exception as e: found[run]={"err":str(e)}
    return found
def metrics(text):
    w=text.split()
    if len(w)<3: return None
    grams=[tuple(w[i:i+3]) for i in range(len(w)-2)]
    cnt=collections.Counter(grams)
    d3=len(cnt)/len(grams); top=cnt.most_common(1)[0][1]
    return len(w),d3,top
rows=[]
for run,info in sorted(runs_with_maxturns1().items()):
    if info.get("max_turns")!=1: continue
    tfs=glob.glob(run+"/round_*/transcripts/*.json")
    d3s=[]; tops=[]; lens=[]; loopy=0; n=0
    for tf in tfs:
        try: msgs=json.load(open(tf))["messages"]
        except: continue
        for m in msgs:
            if m.get("source")=="target":
                mm=metrics((m.get("content") or ""))
                if not mm: continue
                L,d3,top=mm; d3s.append(d3); tops.append(top); lens.append(L); n+=1
                if d3<0.5: loopy+=1
    if n==0: continue
    rows.append((os.path.basename(run),n,statistics.mean(d3s),min(d3s),statistics.mean(tops),max(tops),statistics.mean(lens),loopy,info["corrupt"]))
rows.sort(key=lambda r:r[2])  # lowest mean distinct-3 first = most repetition
print("%-34s %3s %7s %6s %7s %6s %6s %5s corr" % ("run","n","meanD3","minD3","meanTop","maxTop","meanW","loopy"))
for r in rows:
    print(f"{r[0]:34} {r[1]:>3} {r[2]:>7.3f} {r[3]:>6.3f} {r[4]:>7.1f} {r[5]:>6} {r[6]:>6.0f} {r[7]:>5} {r[8]}")
