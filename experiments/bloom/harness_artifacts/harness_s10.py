import sys, json, glob, collections, statistics, torch
sys.path.insert(0, "experiments/bloom")
import bloom_beast_tree_corrupt as B
from transformers import AutoModelForCausalLM, AutoTokenizer
M="Qwen/Qwen3-4B"; DEV="cuda:0"; SEED=42; MAXTOK=250
RUN="experiments/bloom/runs_16/selfcorr_cfg_1t_b6_g1"
tok=AutoTokenizer.from_pretrained(M); EOS=tok.eos_token_id
mt=AutoModelForCausalLM.from_pretrained(M,dtype=torch.bfloat16,attn_implementation="sdpa").to(DEV).eval()
hf={"mt":mt,"mc":mt,"tok":tok,"device":DEV,"pad_id":EOS,"eos_id":EOS}
target_msgs=[]; vids=[]
for tf in sorted(glob.glob(RUN+"/round_1/transcripts/*.json")):
    msgs=json.load(open(tf))["messages"]
    target_msgs.append([{"role":m["role"],"content":m.get("content") or ""} for m in msgs if m.get("role") in ("system","user")])
    vids.append(tf.split("transcript_")[1].split(".")[0])
# s10: ONE corruption prompt (p0), sampled 10x. num_prompts=1, samples_per_prompt=10.
cfg=dict(engine="hf_full",hf=hf,rewrite_prompts=B._DEFAULT_CORRUPTION_REWRITE_PROMPTS,
     num_prompts=1,samples_per_prompt=10,beta=6.0,target_temp=1.0,poe_temp=None,
     min_p=0.0,abs_floor=0.0,target_floor=1e-5,corrupt_only=False,
     cfg_b1=1.0,cfg_b2=12.0,cfg_b3=6.0,cfg_neutral_prompt=None,poe_gen_batch=32,rep_penalty=None)
def met(t):
    w=t.split()
    if len(w)<3: return 1.0,1.0
    g=[tuple(w[i:i+3]) for i in range(len(w)-2)]
    return len(collections.Counter(g))/len(g), len(set(w))/len(w)
torch.manual_seed(SEED)
res=B._corruption_generate_hf(hf,cfg,target_msgs,MAXTOK,1.0,True)
rows=[]
for v,r in zip(vids,res):
    d3,d1=met(r["best_text"]); rows.append({"vid":v,"distinct3":round(d3,3),"distinct1":round(d1,3),"text":r["best_text"]})
json.dump(rows,open("/workspace/out_s10.json","w"))
print("[s10 (p0 x10)] meanD3=%.3f minD3=%.3f loopy=%d | meanD1=%.3f minD1=%.3f"%(statistics.mean([x["distinct3"] for x in rows]),min(x["distinct3"] for x in rows),sum(1 for x in rows if x["distinct3"]<0.5),statistics.mean([x["distinct1"] for x in rows]),min(x["distinct1"] for x in rows)))
print("DONE")
