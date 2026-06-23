import json, glob, os, collections, statistics
BASE="experiments/bloom/runs_16"
def runs():
    found={}
    for cfgp in glob.glob(BASE+"/*/round_*/cfg.json"):
        run=cfgp.split("/round_")[0]
        if run in found: continue
        try:
            c=json.load(open(cfgp)); ro=c.get("rollout") or {}
            found[run]={"max_turns":ro.get("max_turns"),"corrupt":bool((c.get("corruption_output") or {}).get("enabled"))}
        except Exception as e: found[run]={"max_turns":None}
    return found
def metrics(text):
    w=text.split()
    if len(w)<3: return None
    grams=[tuple(w[i:i+3]) for i in range(len(w)-2)]
    cnt=collections.Counter(grams)
    return len(w), len(cnt)/len(grams), cnt.most_common(1)[0][1]
rows=[]
for run,info in runs().items():
    if info.get("max_turns")!=1: continue
    tfs=glob.glob(run+"/round_*/transcripts/*.json")
    d3s=[]; tops=[]; loopy=0; n=0; cdate="?"
    for tf in tfs:
        try: d=json.load(open(tf)); msgs=d["messages"]
        except: continue
        if cdate=="?": cdate=(d.get("metadata") or {}).get("created_at","?")[:10]
        for m in msgs:
            if m.get("source")=="target":
                mm=metrics(m.get("content") or "")
                if not mm: continue
                L,d3,top=mm; d3s.append(d3); tops.append(top); n+=1
                if d3<0.5: loopy+=1
    if n!=25: continue
    rows.append((os.path.basename(run),n,statistics.mean(d3s),min(d3s),statistics.mean(tops),max(tops),loopy,info["corrupt"],cdate))
rows.sort(key=lambda r:r[2])
print("%-26s %3s %7s %6s %7s %6s %5s %4s %10s" % ("run","n","meanD3","minD3","meanTop","maxTop","loopy","cor","created"))
for r in rows:
    print("%-26s %3d %7.3f %6.3f %7.1f %6d %5d %4s %10s" % (r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8]))
