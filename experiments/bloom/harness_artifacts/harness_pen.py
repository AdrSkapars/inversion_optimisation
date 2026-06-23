import sys, json, glob, collections, statistics, torch
sys.path.insert(0, "experiments/bloom")
import bloom_beast_tree_corrupt as B
from transformers import AutoModelForCausalLM, AutoTokenizer

M="Qwen/Qwen3-4B"; DEV="cuda:0"; SEED=42; MAXTOK=250
RUN="experiments/bloom/runs_16/selfcorr_cfg_1t_b6_g1"
# (tag, rep_penalty spec)
SPECS=[("off",None),("freq0.3",{"mode":"freq","alpha_freq":0.3}),
       ("freq0.6",{"mode":"freq","alpha_freq":0.6}),("freq1.0",{"mode":"freq","alpha_freq":1.0})]

tok=AutoTokenizer.from_pretrained(M); EOS=tok.eos_token_id
mt=AutoModelForCausalLM.from_pretrained(M,dtype=torch.bfloat16,attn_implementation="sdpa").to(DEV).eval()
hf={"mt":mt,"mc":mt,"tok":tok,"device":DEV,"pad_id":EOS,"eos_id":EOS}
target_msgs=[]; vids=[]
for tf in sorted(glob.glob(RUN+"/round_1/transcripts/*.json")):
    msgs=json.load(open(tf))["messages"]
    target_msgs.append([{"role":m["role"],"content":m.get("content") or ""} for m in msgs if m.get("role") in ("system","user")])
    vids.append(tf.split("transcript_")[1].split(".")[0])
base_cfg=dict(engine="hf_full",hf=hf,rewrite_prompts=B._DEFAULT_CORRUPTION_REWRITE_PROMPTS,
     num_prompts=10,samples_per_prompt=1,beta=6.0,target_temp=1.0,poe_temp=None,
     min_p=0.0,abs_floor=0.0,target_floor=1e-5,corrupt_only=False,
     cfg_b1=1.0,cfg_b2=12.0,cfg_b3=6.0,cfg_neutral_prompt=None,poe_gen_batch=32)
def met(t):
    w=t.split()
    if len(w)<3: return len(w),1.0,0
    g=[tuple(w[i:i+3]) for i in range(len(w)-2)];c=collections.Counter(g)
    return len(w),len(c)/len(g),c.most_common(1)[0][1]
allres={}
for tag,spec in SPECS:
    cfg=dict(base_cfg); cfg["rep_penalty"]=spec
    torch.manual_seed(SEED)
    res=B._corruption_generate_hf(hf,cfg,target_msgs,MAXTOK,1.0,True)
    rows=[]
    for vid,r in zip(vids,res):
        wl,d3,top=met(r["best_text"]); rows.append({"vid":vid,"words":wl,"distinct3":round(d3,3),"topN":top,"text":r["best_text"]})
    allres[tag]=rows; json.dump(rows,open("/workspace/out_%s.json"%tag,"w"))
    d3s=[x["distinct3"] for x in rows]; loopy=sum(1 for x in rows if x["distinct3"]<0.5)
    print("[%s] meanD3=%.3f minD3=%.3f loopy=%d meanWords=%.0f"%(tag,statistics.mean(d3s),min(d3s),loopy,statistics.mean([x["words"] for x in rows])))
# per-scenario distinct3 across modes for the loopers
print("\nPER-SCENARIO distinct3 (loop-prone):")
tags=[t for t,_ in SPECS]
print("vid     "+"  ".join("%7s"%t for t in tags))
bymode={t:{x["vid"]:x for x in allres[t]} for t in tags}
for vid in ["v14r1","v23r1","v3r1","v8r1","v21r1","v10r1"]:
    print("%-7s "%vid+"  ".join("%7.3f"%bymode[t][vid]["distinct3"] for t in tags))
print("DONE")
