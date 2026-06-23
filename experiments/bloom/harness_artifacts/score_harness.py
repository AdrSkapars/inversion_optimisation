import sys, json, glob, statistics as st, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
M="Qwen/Qwen3-4B"; NO_THINK="<think>\n\n</think>\n"; DEV="cuda:0"
RUN="experiments/bloom/runs_16/selfcorr_cfg_1t_b6_g1"
MODES=[("off","out_off.json"),("freq1.0","out_freq1.0.json"),("ng3l4","out_ng3l4.json"),("dry_b40","out_dry_b40.json")]
tok=AutoTokenizer.from_pretrained(M)
mt=AutoModelForCausalLM.from_pretrained(M,dtype=torch.bfloat16,attn_implementation="sdpa").to(DEV).eval()
# prefixes per vid (system+user)
pre={}
for tf in glob.glob(RUN+"/round_1/transcripts/*.json"):
    vid=tf.split("transcript_")[1].split(".")[0]
    msgs=json.load(open(tf))["messages"]
    conv=[{"role":m["role"],"content":m.get("content") or ""} for m in msgs if m.get("role") in ("system","user")]
    s=tok.apply_chat_template(conv,tokenize=False,add_generation_prompt=True)+NO_THINK
    pre[vid]=tok.encode(s,add_special_tokens=False)
def score(text,t_pre):
    rid=tok.encode(text,add_special_tokens=False)
    if not rid: return None
    full=t_pre+rid
    with torch.no_grad():
        lp=torch.log_softmax(mt(input_ids=torch.tensor([full],device=DEV)).logits[0].float(),-1)
        cols=torch.arange(len(t_pre)-1,len(full)-1,device=DEV); tg=torch.tensor(rid,device=DEV)
        p=lp[cols].gather(-1,tg.unsqueeze(-1)).squeeze(-1).exp()
    return [float(x)*100 for x in p]
out={}
for tag,f in MODES:
    rows={o["vid"]:o for o in json.load(open("/workspace/"+f))}
    allp=[]; omins=[]
    for vid,t_pre in pre.items():
        pl=score(rows[vid]["text"],t_pre)
        if pl: allp+=pl; omins.append(min(pl))
    summ=dict(A_mean=sum(allp)/len(allp),A_median=st.median(allp),B_min_of_mins=min(omins),B_mean_of_mins=sum(omins)/len(omins))
    out[tag]=summ
    print("[%-7s] avg-prob(A_mean)=%.2f%%  A_median=%.2f%%  least-token(B_min)=%.6f%%  B_mean_of_mins=%.4f%%"%(tag,summ["A_mean"],summ["A_median"],summ["B_min_of_mins"],summ["B_mean_of_mins"]))
json.dump(out,open("/workspace/score_summary.json","w"),indent=1)
print("saved")
