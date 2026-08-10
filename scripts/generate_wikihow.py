#!/usr/bin/env python3
import json, os, re, time, random, argparse, logging
from pathlib import Path
from datetime import datetime
import numpy as np

TEXT_ROOT = Path.home() / "Text"
M4_DATA   = TEXT_ROOT / "data/raw/m4"
OUT_DIR   = TEXT_ROOT / "data/raw/wikihow_generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_WORDS  = 100
MAX_WORDS  = 600
MIN_CHARS  = 500
TEMP       = 0.7
MAX_TOKENS = 900

REJECT_RE = re.compile(
    r"(^(i cannot|i can\'t|i am unable|as an ai|i\'m sorry)"
    r"|(i cannot (create|write|generate|provide))"
    r"|(note:|disclaimer:)\s+this"
    r"|wikihow article)", re.IGNORECASE)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def extract_m4_prompts():
    prompts = {}
    for fp in sorted(M4_DATA.glob("wikihow_*.jsonl")):
        with open(fp, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: row = json.loads(line)
                except: continue
                pt = row.get("prompt",""); ht = row.get("human_text","")
                m = re.search(r"from title \'([^\']+)\'", pt)
                if not m or not pt: continue
                title = m.group(1).strip()
                if not title or title in prompts: continue
                if len(ht.split()) < 50: continue
                skip_kw = ["suppressor","silencer","weapon","explosive",
                    "poison","bomb","drug","hack","malware","suicide",
                    "self-harm","pick a lock","hotwire"]
                if any(k in title.lower() for k in skip_kw): continue
                prompts[title] = {"title":title,"prompt":pt,"human_text":ht}
    logger.info(f"Extracted {len(prompts):,} unique prompts")
    return list(prompts.values())

def quality_check(text):
    if not text: return False,"empty"
    text = text.strip()
    if len(text.split()) < MIN_WORDS: return False,f"short_{len(text.split())}w"
    if len(text) < MIN_CHARS: return False,f"short_{len(text)}c"
    if REJECT_RE.search(text[:300]): return False,"refusal"
    if text.count(".") < 3 and "\n" not in text: return False,"no_structure"
    return True,"ok"

def clean(text):
    m = re.match(r"^(sure[!,]?\s+here[^.]*\.\s*|here is[^.]*\.\s*)", text, re.IGNORECASE)
    if m: text = text[m.end():]
    words = text.split()
    if len(words) > MAX_WORDS:
        t = " ".join(words[:MAX_WORDS]); lp = t.rfind(".")
        text = t[:lp+1] if lp > len(t)*0.7 else t
    return text.strip()

def rec(item, text, model, gid):
    return {"title":item["title"],"prompt":item["prompt"],
            "machine_text":clean(text),"human_text":item["human_text"],
            "model":model,"source":"wikihow","generator_id":gid,
            "temperature":TEMP,"word_count":len(text.split()),
            "generated_at":datetime.now().isoformat()}

def save(results, gid):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / f"wikihow_{gid}_{len(results)}samples_{ts}.jsonl"
    with open(out,"w",encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    logger.info(f"Saved {len(results):,} → {out}")
    if results:
        wcs=[r["word_count"] for r in results]
        logger.info(f"Words: mean={np.mean(wcs):.0f} median={np.median(wcs):.0f} min={min(wcs)} max={max(wcs)}")
    return out

# Append single result immediately to disk
_open_files = {}
def save_one(r, gid):
    if gid not in _open_files:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = OUT_DIR / f"wikihow_{gid}_streaming_{ts}.jsonl"
        _open_files[gid] = open(fp, "w", encoding="utf-8")
        logger.info(f"Streaming to: {fp}")
    _open_files[gid].write(json.dumps(r, ensure_ascii=False)+"\n")
    _open_files[gid].flush()

def gen_openai(sel, key):
    from openai import OpenAI
    c=OpenAI(api_key=key); m="gpt-4o"; res=[]
    for item in sel:
        try:
            r=c.chat.completions.create(model=m,messages=[{"role":"user","content":item["prompt"]}],max_tokens=MAX_TOKENS,temperature=TEMP)
            t=r.choices[0].message.content or ""
            ok,_=quality_check(t)
            if ok:
                res.append(rec(item,t,m,"gpt4o"))
                save_one(res[-1],"gpt4o")
            time.sleep(0.1)
        except Exception as e: logger.warning(f"OpenAI: {e}"); time.sleep(2)
    return res

def gen_cohere(sel, key):
    import cohere
    c=cohere.ClientV2(api_key=key); m="command-a-03-2025"; res=[]
    for item in sel:
        try:
            r=c.chat(model=m,messages=[{"role":"user","content":item["prompt"]}],max_tokens=MAX_TOKENS,temperature=TEMP)
            t=r.message.content[0].text or ""
            ok,_=quality_check(t)
            if ok:
                res.append(rec(item,t,m,"cohere"))
                save_one(res[-1],"cohere")
            time.sleep(0.1)
        except Exception as e: logger.warning(f"Cohere: {e}"); time.sleep(2)
    return res

def gen_gemini(sel, key):
    from google import genai as google_genai
    from google.genai import types as gtypes
    client = google_genai.Client(api_key=key)
    m = "gemini-3.5-flash"; res = []
    for item in sel:
        try:
            gemini_prompt = (
                item["prompt"] +
                "\n\nWrite a COMPLETE wikihow article with ALL steps fully explained. "
                "Each step must have 3-5 sentences. Write at least 400 words total. "
                "Do not truncate or summarize — write the full article."
            )
            r = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=gemini_prompt,
                config=gtypes.GenerateContentConfig(
                    max_output_tokens=2048,
                    temperature=TEMP,
                )
            )
            t = (r.text or "").encode("utf-8","replace").decode("utf-8")
            words = len(t.split())
            if words < 100:
                logger.warning(f"Gemini too short: {words} words — skipping")
            ok,_ = quality_check(t)
            if ok:
                res.append(rec(item,t,m,"gemini"))
                save_one(res[-1],"gemini")
                logger.info(f"Gemini saved: {words} words")
            time.sleep(0.15)
        except Exception as e:
            logger.warning(f"Gemini: {e}")
            wait = 30 if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) else 2
            time.sleep(wait)
    return res

def gen_anthropic(sel, key):
    import anthropic
    c=anthropic.Anthropic(api_key=key); m="claude-haiku-4-5"; res=[]
    for item in sel:
        try:
            r=c.messages.create(model=m,max_tokens=MAX_TOKENS,temperature=TEMP,messages=[{"role":"user","content":item["prompt"]}])
            t=r.content[0].text or ""
            ok,_=quality_check(t)
            if ok:
                res.append(rec(item,t,m,"claude"))
                save_one(res[-1],"claude")
            time.sleep(0.1)
        except Exception as e: logger.warning(f"Anthropic: {e}"); time.sleep(2)
    return res

def gen_groq(sel, key, gid="llama70b"):
    from groq import Groq
    c=Groq(api_key=key)
    mm={"llama70b":("llama-3.3-70b-versatile","llama-3.3-70b"),
        "qwen32b":("qwen/qwen3-32b","qwen3-32b"),
        "llama8b":("meta-llama/llama-3.1-8b-instant","llama-3.1-8b"),
        "mistral":("meta-llama/llama-4-scout-17b-16e-instruct","llama-4-scout-17b")}
    ms,mn=mm.get(gid,mm["llama70b"]); res=[]
    for item in sel:
        try:
            r=c.chat.completions.create(model=ms,messages=[{"role":"user","content":item["prompt"]}],max_tokens=MAX_TOKENS,temperature=TEMP)
            t=r.choices[0].message.content or ""
            ok,_=quality_check(t)
            if ok:
                res.append(rec(item,t,mn,gid))
                save_one(res[-1],gid)
            time.sleep(0.05)
        except Exception as e:
            logger.warning(f"Groq[{gid}]: {e}")
            wait = 65 if "429" in str(e) or "rate_limit" in str(e) else 1
            if wait > 1: logger.info(f"  Rate limited — waiting {wait}s...")
            time.sleep(wait)
    return res

def gen_grok(sel, key):
    from openai import OpenAI
    c=OpenAI(api_key=key,base_url="https://api.x.ai/v1"); m="grok-4.5"; res=[]
    for item in sel:
        try:
            r=c.chat.completions.create(model=m,messages=[{"role":"user","content":item["prompt"]}],max_tokens=MAX_TOKENS,temperature=TEMP)
            t=r.choices[0].message.content or ""
            ok,_=quality_check(t)
            if ok:
                res.append(rec(item,t,m,"grok"))
                save_one(res[-1],"grok")
            time.sleep(0.1)
        except Exception as e: logger.warning(f"Grok: {e}"); time.sleep(2)
    return res


def gen_hf_local(sel, model_id, gid):
    """Run any HuggingFace model locally on GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch
    logger.info(f"Loading {model_id} in 4-bit on GPU...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb,
        device_map="auto", trust_remote_code=True)
    mdl.eval()
    logger.info(f"✓ {model_id} loaded")
    res = []
    for item in sel:
        try:
            msgs = [{"role":"user","content":item["prompt"]}]
            txt_in = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tok(txt_in, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
            with torch.no_grad():
                out = mdl.generate(**inp, max_new_tokens=MAX_TOKENS,
                                   temperature=TEMP, do_sample=True,
                                   pad_token_id=tok.eos_token_id)
            t = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            ok,_ = quality_check(t)
            if ok:
                res.append(rec(item, t, model_id.split("/")[-1], gid))
                save_one(res[-1], gid)
        except Exception as e:
            logger.warning(f"HF[{gid}]: {e}")
    return res


def load_done_titles(gid):
    import glob as _glob
    done = set()
    for fp in _glob.glob(str(OUT_DIR / f"wikihow_{gid}_*.jsonl")):
        try:
            with open(fp, encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        if r.get("generator_id") == gid:
                            done.add(r["title"])
                    except: pass
        except: pass
    logger.info(f"[{gid}] Already done: {len(done):,} titles — will skip these")
    return done

GENS={"gpt4o":"GPT-4o","cohere":"Command R+","gemini":"Gemini-2.0-Flash",
      "claude":"Claude-3.5-Haiku","llama70b":"Llama-3.1-70B (Groq)",
      "llama8b":"Llama-3.1-8B (Groq)","mistral":"Mistral-Saba (Groq)",
      "grok":"Grok-4.5","qwen":"Qwen-2.5-72B (local)","qwen32b":"Qwen3-32B (Groq, free)",
    "mistral7b":"Mistral-7B-Instruct (local)","qwen14b":"Qwen2.5-14B-Instruct (local)","llama8b_local":"Llama-3.1-8B-Instruct (local)"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--generator",required=True,choices=list(GENS.keys()))
    p.add_argument("--n",type=int,default=2000)
    p.add_argument("--openai-key",default=os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--cohere-key",default=os.environ.get("COHERE_API_KEY"))
    p.add_argument("--gemini-key",default=os.environ.get("GEMINI_API_KEY"))
    p.add_argument("--claude-key",default=os.environ.get("ANTHROPIC_API_KEY"))
    p.add_argument("--groq-key",  default=os.environ.get("GROQ_API_KEY"))
    p.add_argument("--grok-key",  default=os.environ.get("XAI_API_KEY"))
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()

    print(f"\n{'='*60}")
    print(f"  WikiHow Generation — {GENS[a.generator]}")
    print(f"  Target={a.n:,}  Temp={TEMP}  MaxTokens={MAX_TOKENS}")
    print(f"  Quality: >={MIN_WORDS}w / >={MIN_CHARS}c / no refusals")
    print(f"{'='*60}\n")

    all_p=extract_m4_prompts()
    random.seed(42); random.shuffle(all_p)
    sel=all_p[:a.n]
    done_titles = load_done_titles(a.generator)
    if done_titles:
        before = len(sel)
        sel = [p for p in sel if p["title"] not in done_titles]
        logger.info(f"Skipping {before-len(sel):,} done, {len(sel):,} new to generate")
    print(f"New to generate: {len(sel):,} (already done: {len(done_titles):,})\n")

    if a.dry_run:
        s=sel[0]
        print(f"DRY RUN\nTitle:  {s['title']}\nPrompt: {s['prompt'][:300]}...\nHuman:  {len(s['human_text'].split())} words")
        return

    g=a.generator; res=[]
    if g=="gpt4o":    res=gen_openai(sel,a.openai_key)
    elif g=="cohere": res=gen_cohere(sel,a.cohere_key)
    elif g=="gemini": res=gen_gemini(sel,a.gemini_key)
    elif g=="claude": res=gen_anthropic(sel,a.claude_key)
    elif g in("llama70b","llama8b","mistral","qwen32b"): res=gen_groq(sel,a.groq_key,g)
    elif g=="grok":   res=gen_grok(sel,a.grok_key)
    elif g=="mistral7b":    res=gen_hf_local(sel,"mistralai/Mistral-7B-Instruct-v0.3","mistral7b")
    elif g=="qwen14b":      res=gen_hf_local(sel,"Qwen/Qwen2.5-14B-Instruct","qwen14b")
    elif g=="llama8b_local":res=gen_hf_local(sel,"meta-llama/Llama-3.1-8B-Instruct","llama8b_local")

    if res: save(res,g)
    else: print("No results — check API keys and logs")

if __name__=="__main__": main()
