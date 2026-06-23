import sys, json, collections, torch
sys.path.insert(0, "experiments/bloom")
import bloom_beast_tree_corrupt as B
from transformers import AutoModelForCausalLM, AutoTokenizer

M="Qwen/Qwen3-4B"; DEV="cuda:0"; NO_THINK="<think>\n\n</think>\n"
R="experiments/bloom/runs_16/selfcorr_cfg_3t_b6_g1"
PIDX=8
# --- run config (from cfg.json, legacy cfg_gamma path) ---
BETA=6.0; CFG_GAMMA=1.0; TARGET_TEMP=1.0; TARGET_FLOOR=1e-5; POE_TEMP=1.0; MAXTOK=250
SEEDS=list(range(16))

tok=AutoTokenizer.from_pretrained(M); EOS=tok.eos_token_id; PAD=EOS
mt=AutoModelForCausalLM.from_pretrained(M,dtype=torch.bfloat16,attn_implementation="sdpa").to(DEV).eval()
NEUTRAL=B._DEFAULT_CFG_NEUTRAL_PROMPT; PROMPTS=B._DEFAULT_CORRUPTION_REWRITE_PROMPTS

# turn-2 context = messages[0:4] (sys,user1,assistant1_corrupted,user2)
msgs=json.load(open(R+"/round_1/transcripts/transcript_v23r1.json"))["messages"]
conv=[{"role":m["role"],"content":m.get("content") or ""} for m in msgs[:4]]
ts=tok.apply_chat_template(conv,tokenize=False,add_generation_prompt=True)+NO_THINK
t_pre=tok.encode(ts,add_special_tokens=False)

# --- faithful: sample baseline per seed (temp 1.0) then PoE, like the pipeline ---
import collections
def degen(ids):
    ids=[x for x in ids if x!=EOS]; L=len(ids); term=(L<MAXTOK)
    grams=[tuple(ids[i:i+3]) for i in range(max(0,L-2))]; cnt=collections.Counter(grams)
    d3=(len(cnt)/len(grams)) if grams else 1.0
    top,tc=(cnt.most_common(1)[0] if cnt else ((),0))
    return L,term,d3,tc,tok.decode(list(top))
print(" seed  len  term  distinct3  topGramN  topgram | base-head -> tail")
for sd in SEEDS:
    torch.manual_seed(sd)
    base_ids=B._hf_generate(mt,[list(t_pre)],MAXTOK,1.0,PAD,EOS,DEV)[0]
    baseline=tok.decode([x for x in base_ids if x!=EOS],skip_special_tokens=True).strip()
    cs=tok.apply_chat_template(B.build_corruption_msgs(PROMPTS[PIDX],baseline),tokenize=False,add_generation_prompt=True)+NO_THINK
    ns=tok.apply_chat_template(B.build_corruption_msgs(NEUTRAL,baseline),tokenize=False,add_generation_prompt=True)+NO_THINK
    c_pre=tok.encode(cs,add_special_tokens=False); n_pre=tok.encode(ns,add_special_tokens=False)
    gen=B._hf_poe_generate(mt,mt,[list(t_pre)],[list(c_pre)],BETA,POE_TEMP,MAXTOK,PAD,EOS,DEV,
                           target_temp=TARGET_TEMP,target_floor=TARGET_FLOOR,cfg_gamma=CFG_GAMMA,n_prefs=[list(n_pre)])[0]
    L,term,d3,tc,tg=degen(gen)
    txt=tok.decode([x for x in gen if x!=EOS],skip_special_tokens=True)
    flag="<<LOOP" if (d3<0.6 or tc>=10) else ""
    print(f"{sd:>5} {L:>4} {str(term):>5} {d3:>9.3f} {tc:>8}  [{tg!r}] | ...{txt[-58:].strip()!r} {flag}")
