import sys, json, glob, collections, statistics, torch
sys.path.insert(0, "experiments/bloom")
import bloom_beast_tree_corrupt as B
from transformers import AutoModelForCausalLM, AutoTokenizer
M="Qwen/Qwen3-4B"; DEV="cuda:0"; SEED=42; MAXTOK=250
RUN="experiments/bloom/runs_16/selfcorr_cfg_1t_b6_g1"
SPECS=[("dry_b13",{"mode":"dry","multiplier":0.8,"base":1.3,"allowed_length":2}),("dry_b25",{"mode":"dry","multiplier":0.8,"base":2.5,"allowed_length":2}),("dry_b40",{"mode":"dry","multiplier":0.8,"base":4.0,"allowed_length":2})]
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
for tag,spec in SPECS:
    cfg=dict(base_cfg); cfg["rep_penalty"]=spec
    torch.manual_seed(SEED)
    res=B._corruption_generate_hf(hf,cfg,target_msgs,MAXTOK,1.0,True)
    rows=[{"vid":v,"words":met(r["best_text"])[0],"distinct3":round(met(r["best_text"])[1],3),"topN":met(r["best_text"])[2],"text":r["best_text"]} for v,r in zip(vids,res)]
    json.dump(rows,open("/workspace/out_%s.json"%tag,"w"))
    d3s=[x["distinct3"] for x in rows]; loopy=sum(1 for x in rows if x["distinct3"]<0.5)
    print("[%s] meanD3=%.3f minD3=%.3f loopy=%d meanWords=%.0f"%(tag,statistics.mean(d3s),min(d3s),loopy,statistics.mean([x["words"] for x in rows])))
print("DONE")
