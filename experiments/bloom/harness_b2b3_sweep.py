import sys, json, glob, collections, statistics, torch
sys.path.insert(0, "experiments/bloom")
import bloom_beast_tree_corrupt as B
from transformers import AutoModelForCausalLM, AutoTokenizer
M="Qwen/Qwen3-4B"; DEV="cuda:0"; SEED=42; MAXTOK=250; NO_THINK="<think>\n\n</think>\n"
RUN="experiments/bloom/runs_16/selfcorr_cfg_1t_b6_g1"
FLOOR=1e-5
# (tag, b2, b3)
POINTS=[("b12_6",12,6),("b8_4",8,4),("b6_3",6,3),("b4_2",4,2),
        ("b12_0",12,0),("b8_0",8,0),("b6_0",6,0),("b4_0",4,0),("b3_0",3,0)]
tok=AutoTokenizer.from_pretrained(M); EOS=tok.eos_token_id
mt=AutoModelForCausalLM.from_pretrained(M,dtype=torch.bfloat16,attn_implementation="sdpa").to(DEV).eval()
hf={"mt":mt,"mc":mt,"tok":tok,"device":DEV,"pad_id":EOS,"eos_id":EOS}
VOCAB=float(getattr(mt.config,"vocab_size",151936)); IMPOSS=100.0/VOCAB; FLOORPCT=FLOOR*100.0
target_msgs=[]; vids=[]; t_pres=[]
for tf in sorted(glob.glob(RUN+"/round_1/transcripts/*.json")):
    msgs=json.load(open(tf))["messages"]
    conv=[{"role":m["role"],"content":m.get("content") or ""} for m in msgs if m.get("role") in ("system","user")]
    target_msgs.append(conv); vids.append(tf.split("transcript_")[1].split(".")[0])
    s=tok.apply_chat_template(conv,tokenize=False,add_generation_prompt=True)+NO_THINK
    t_pres.append(tok.encode(s,add_special_tokens=False))
base=dict(engine="hf_full",hf=hf,rewrite_prompts=B._DEFAULT_CORRUPTION_REWRITE_PROMPTS,
     num_prompts=1,samples_per_prompt=10,beta=6.0,target_temp=1.0,poe_temp=None,
     min_p=0.0,abs_floor=0.0,target_floor=FLOOR,corrupt_only=False,
     cfg_b1=1.0,cfg_neutral_prompt=None,poe_gen_batch=32,rep_penalty=None)
def rep(t):
    w=t.split()
    if len(w)<3: return 1.0,1.0
    g=[tuple(w[i:i+3]) for i in range(len(w)-2)]
    return len(collections.Counter(g))/len(g), len(set(w))/len(w)
def score(text,t_pre):
    rid=tok.encode(text,add_special_tokens=False)
    if not rid: return [],None,0
    full=t_pre+rid
    with torch.no_grad():
        lp=torch.log_softmax(mt(input_ids=torch.tensor([full],device=DEV)).logits[0].float(),-1)
        cols=torch.arange(len(t_pre)-1,len(full)-1,device=DEV); tg=torch.tensor(rid,device=DEV)
        p=lp[cols].gather(-1,tg.unsqueeze(-1)).squeeze(-1).exp()
    pl=[float(x)*100 for x in p]
    nimp=sum(1 for x in pl if x<IMPOSS)
    return pl, min(pl), nimp
print("tag      b2 b3 | meanD3 minD3 loopy | meanD1 | avg-prob | least-tok(clip) | imposs")
allres={}
for tag,b2,b3 in POINTS:
    cfg=dict(base); cfg["cfg_b2"]=float(b2); cfg["cfg_b3"]=float(b3)
    torch.manual_seed(SEED)
    res=B._corruption_generate_hf(hf,cfg,target_msgs,MAXTOK,1.0,True)
    d3s=[];d1s=[];allp=[];omins=[];imp=0;rows=[]
    for vid,r,t_pre in zip(vids,res,t_pres):
        t=r["best_text"]; d3,d1=rep(t); d3s.append(d3); d1s.append(d1)
        pl,mn,ni=score(t,t_pre); allp+=pl; 
        if mn is not None: omins.append(mn)
        imp+=ni; rows.append({"vid":vid,"d3":round(d3,3),"d1":round(d1,3),"text":t})
    allres[tag]=rows; json.dump(rows,open("/workspace/sweep_%s.json"%tag,"w"))
    avgprob=sum(allp)/len(allp); lt=min(max(x,FLOORPCT) for x in omins)
    print("%-8s %2d %2d | %.3f  %.3f   %d   | %.3f | %6.2f%% | %.5f%%      | %d"%(tag,b2,b3,statistics.mean(d3s),min(d3s),sum(1 for x in d3s if x<0.5),statistics.mean(d1s),avgprob,lt,imp))
print("SWEEPDONE")
