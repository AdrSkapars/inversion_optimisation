import sys, json, glob, collections, torch
sys.path.insert(0,"experiments/bloom")
import bloom_beast_tree_corrupt as B
from transformers import AutoModelForCausalLM, AutoTokenizer
M="Qwen/Qwen3-4B"; DEV="cuda:0"; SEED=42
RUN="experiments/bloom/runs_16/selfcorr_cfg_1t_b6_g1"
tok=AutoTokenizer.from_pretrained(M); EOS=tok.eos_token_id
mt=AutoModelForCausalLM.from_pretrained(M,dtype=torch.bfloat16,attn_implementation="sdpa").to(DEV).eval()
hf={"mt":mt,"mc":mt,"tok":tok,"device":DEV,"pad_id":EOS,"eos_id":EOS}
tf=glob.glob(RUN+"/round_1/transcripts/transcript_v23r1.json")[0]
msgs=json.load(open(tf))["messages"]
conv=[{"role":m["role"],"content":m.get("content") or ""} for m in msgs if m.get("role") in ("system","user")]
base=dict(engine="hf_full",hf=hf,rewrite_prompts=B._DEFAULT_CORRUPTION_REWRITE_PROMPTS,num_prompts=10,samples_per_prompt=1,
     beta=6.0,target_temp=1.0,poe_temp=None,min_p=0.0,abs_floor=0.0,target_floor=1e-5,corrupt_only=False,
     cfg_b1=1.0,cfg_b2=12.0,cfg_b3=6.0,cfg_neutral_prompt=None,poe_gen_batch=32)
def d3(t):
    w=t.split()
    if len(w)<3: return 1.0
    g=[tuple(w[i:i+3]) for i in range(len(w)-2)];c=collections.Counter(g);return len(c)/len(g)
for tag,spec in [("off",None),("ng4l8",{"mode":"ngram","n":4,"lam":8})]:
    cfg=dict(base); cfg["rep_penalty"]=spec
    torch.manual_seed(SEED)
    res=B._corruption_generate_hf(hf,cfg,[conv],250,1.0,True)[0]
    print("==== %s ==== best_text d3=%.3f"%(tag,d3(res["best_text"])))
    pool=sorted(res["pool"],key=lambda x:(x["target_lp"] if x["target_lp"] is not None else -9), reverse=True)
    print("rank pidx  target_lp  target_p%%  d3    tail")
    for r,c in enumerate(pool):
        lp=c["target_lp"]; pp=c.get("target_p_pct")
        print("  %d   p%-2d  %8.3f  %7.2f  %.3f  ...%s"%(r,c["prompt_index"],lp if lp else 0,pp if pp else 0,d3(c["text"]),c["text"][-44:].replace(chr(10)," ")))
