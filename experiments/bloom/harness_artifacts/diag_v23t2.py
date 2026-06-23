import json, ast, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
M="Qwen/Qwen3-4B"; NO_THINK="<think>\n\n</think>\n"; DEV="cuda:0"
R="experiments/bloom/runs_16/selfcorr_cfg_3t_b6_g1"
PIDX=8  # turn-2 selected prompt_index
src=open("experiments/bloom/bloom_beast_tree_corrupt.py").read()
def getlit(name):
    for n in ast.parse(src).body:
        if isinstance(n,ast.Assign) and any(getattr(t,"id",None)==name for t in n.targets):
            return ast.literal_eval(n.value)
PROMPTS=getlit("_DEFAULT_CORRUPTION_REWRITE_PROMPTS"); NEUTRAL=getlit("_DEFAULT_CFG_NEUTRAL_PROMPT")
def build(instr,base): return [{"role":"user","content":instr.rstrip()+"\n\n"+base}]
d=json.load(open(R+"/round_1/transcripts/transcript_v23r1.json"))
msgs=d["messages"]
# context through turn 2 user = messages 0..3 (system,user1,assistant1_corrupted,user2)
conv=[{"role":m["role"],"content":m.get("content") or ""} for m in msgs[:4]]
out_text=msgs[4]["content"] or ""  # turn-2 corrupted output
tok=AutoTokenizer.from_pretrained(M); EOS=tok.eos_token_id
mt=AutoModelForCausalLM.from_pretrained(M,dtype=torch.bfloat16,attn_implementation="sdpa").to(DEV).eval()
ts=tok.apply_chat_template(conv,tokenize=False,add_generation_prompt=True)+NO_THINK
tids=tok(ts,return_tensors="pt").to(DEV)
with torch.no_grad():
    bo=mt.generate(**tids,max_new_tokens=220,do_sample=False,pad_token_id=tok.eos_token_id)
baseline=tok.decode(bo[0][tids.input_ids.shape[1]:],skip_special_tokens=True).strip()
pres={"target":ts,
 "corrupt":tok.apply_chat_template(build(PROMPTS[PIDX],baseline),tokenize=False,add_generation_prompt=True)+NO_THINK,
 "neutral":tok.apply_chat_template(build(NEUTRAL,baseline),tokenize=False,add_generation_prompt=True)+NO_THINK}
out_ids=tok.encode(out_text,add_special_tokens=False)
def sc(prestr):
    pre=tok.encode(prestr,add_special_tokens=False); full=pre+out_ids
    with torch.no_grad():
        lp=torch.log_softmax(mt(input_ids=torch.tensor([full],device=DEV)).logits[0].float(),-1)
        cols=torch.arange(len(pre)-1,len(full)-1,device=DEV); tg=torch.tensor(out_ids,device=DEV)
        actual=lp[cols].gather(-1,tg.unsqueeze(-1)).squeeze(-1).exp()
        eos=lp[cols][:,EOS].exp()
    return [float(x) for x in actual],[float(x) for x in eos]
res={"prompt_index":PIDX,"tokens":[tok.decode([i]) for i in out_ids]}
for k,pre in pres.items():
    a,e=sc(pre); res[k]=a; res[k+"_eos"]=e
print("DIAGJSON "+json.dumps(res))
