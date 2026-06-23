import json,glob,collections,statistics,os
def reps(run):
    d3s=[];loopy=0;n=0;tops=[]
    for tf in glob.glob(run+"/round_1/transcripts/*.json"):
        for m in json.load(open(tf))["messages"]:
            if m.get("source")=="target":
                w=(m.get("content") or "").split()
                if len(w)<3: continue
                g=[tuple(w[i:i+3]) for i in range(len(w)-2)];c=collections.Counter(g)
                d3=len(c)/len(g);d3s.append(d3);tops.append(c.most_common(1)[0][1]);n+=1
                if d3<0.5: loopy+=1
    return n,statistics.mean(d3s),min(d3s),statistics.mean(tops),max(tops),loopy
for tag,run in [("seed42","experiments/bloom/runs_16/selfcorr_cfg_1t_b6_g1"),("seed1234","experiments/bloom/runs_16/selfcorr_cfg_1t_b6_g1_seed1234")]:
    n,md3,mind3,mtop,maxtop,lp=reps(run)
    st=run+"/round_1/score_tokens.json"; has="yes" if os.path.exists(st) else "MISSING"
    print("%s: n=%d meanD3=%.3f minD3=%.3f meanTop=%.1f maxTop=%d loopy=%d | score_tokens=%s"%(tag,n,md3,mind3,mtop,maxtop,lp,has))
    if os.path.exists(st):
        print("   score_tokens:", json.dumps(json.load(open(st)))[:400])
