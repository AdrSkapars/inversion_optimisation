import sys, json, glob, collections, statistics, torch
sys.path.insert(0, "experiments/bloom")
import bloom_beast_tree_corrupt as B
from transformers import AutoModelForCausalLM, AutoTokenizer

M="Qwen/Qwen3-4B"; DEV="cuda:0"
RUN="experiments/bloom/runs_16/selfcorr_cfg_1t_b6_g1"
SEED=42; MAXTOK=250

tok=AutoTokenizer.from_pretrained(M); EOS=tok.eos_token_id
mt=AutoModelForCausalLM.from_pretrained(M,dtype=torch.bfloat16,attn_implementation="sdpa").to(DEV).eval()
hf={"mt":mt,"mc":mt,"tok":tok,"device":DEV,"pad_id":EOS,"eos_id":EOS}

# fixed inputs: system+user from each transcript (no evaluator needed)
target_msgs=[]; vids=[]
for tf in sorted(glob.glob(RUN+"/round_1/transcripts/*.json")):
    msgs=json.load(open(tf))["messages"]
    conv=[{"role":m["role"],"content":m.get("content") or ""} for m in msgs if m.get("role") in ("system","user")]
    target_msgs.append(conv); vids.append(tf.split("transcript_")[1].split(".")[0])

cfg={"engine":"hf_full","hf":hf,"rewrite_prompts":B._DEFAULT_CORRUPTION_REWRITE_PROMPTS,
     "num_prompts":10,"samples_per_prompt":1,"beta":6.0,"target_temp":1.0,"poe_temp":None,
     "min_p":0.0,"abs_floor":0.0,"target_floor":1e-5,"corrupt_only":False,
     "cfg_b1":1.0,"cfg_b2":12.0,"cfg_b3":6.0,"cfg_neutral_prompt":None,"poe_gen_batch":32}

print("scenarios:",len(target_msgs),"| generating (best-of-10 target-pick, CFG b=1/12/6, seed",SEED,")...")
torch.manual_seed(SEED)
res=B._corruption_generate_hf(hf,cfg,target_msgs,MAXTOK,1.0,True)

def metrics(text):
    w=text.split()
    if len(w)<3: return len(w),1.0,0
    g=[tuple(w[i:i+3]) for i in range(len(w)-2)];c=collections.Counter(g)
    return len(w),len(c)/len(g),c.most_common(1)[0][1]
out=[]
print("%-8s %5s %8s %7s  text-tail"%("vid","words","distinct3","topN"))
for vid,r in zip(vids,res):
    t=r["best_text"]; wl,d3,top=metrics(t)
    out.append({"vid":vid,"words":wl,"distinct3":round(d3,3),"topN":top,"text":t})
    flag="  <<LOOP" if d3<0.5 else ""
    print("%-8s %5d %8.3f %7d  ...%s%s"%(vid,wl,d3,top,t[-46:].replace(chr(10)," "),flag))
d3s=[o["distinct3"] for o in out]; loopy=sum(1 for o in out if o["distinct3"]<0.5)
print("="*70)
print("AGG: n=%d meanD3=%.3f minD3=%.3f loopy(<0.5)=%d  meanWords=%.0f"%(len(out),statistics.mean(d3s),min(d3s),loopy,statistics.mean([o["words"] for o in out])))
json.dump(out,open("/workspace/baseline_outputs.json","w"),indent=1)
print("saved /workspace/baseline_outputs.json")
