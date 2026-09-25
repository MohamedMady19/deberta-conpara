"""62-feature linguistic and statistical extractor for DeBERTa-ConPara.

Extracted verbatim from the training pipeline so that inference,
evaluation and the public demo share one implementation.

Dependencies: numpy, scipy, torch, transformers (GPT-2 for perplexity).
"""
import os, re, gc, zlib, math, string, unicodedata
from collections import Counter, defaultdict

import numpy as np
import torch
from scipy import stats
from scipy.stats import entropy as scipy_entropy
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FeatureExtractor:
    function_words=set(['the','a','an','and','or','but','if','then','else','when','at','by','for','with','about','against','between','into','through','during','before','after','above','below','from','up','down','in','out','on','off','over','under','to','of','is','are','was','were','be','been','being','have','has','had','do','does','did','will','would','could','should','may','might','shall','can','need','dare','ought','used','it','its','this','that','these','those','i','you','he','she','we','they','me','him','her','us','them','my','your','his','our','their','what','which','who','whom','whose','how','where','why','not','no','nor','so','yet','both','either','neither','each','every','all','any','few','more','most','some','such','than','too','very','just','also','as','well','even','still','already'])
    pronouns=set(['i','me','my','myself','you','your','yourself','he','him','his','himself','she','her','hers','herself','it','its','itself','we','us','our','ourselves','they','them','their','themselves'])
    conjunctions=set(['and','but','or','nor','for','yet','so','although','because','since','unless','until','while','after','before','if','though','when','where'])
    def __init__(self,device=None):
        self.device=device or DEVICE; self.gpt2_model=self.gpt2_tokenizer=None
    def _load_gpt2(self):
        if self.gpt2_model is None:
            print("  Loading GPT-2..."); self.gpt2_tokenizer=GPT2TokenizerFast.from_pretrained('gpt2')
            self.gpt2_model=GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
            self.gpt2_model.eval(); self.gpt2_tokenizer.pad_token=self.gpt2_tokenizer.eos_token
    def _tok(self,t): return re.findall(r'\b[a-zA-Z]+\b',t.lower())
    def _sent(self,t): return [s.strip() for s in re.split(r'[.!?]+',t) if s.strip()]
    def _syl(self,w):
        w=w.lower()
        if len(w)<=3: return 1
        c,p=0,False
        for ch in w:
            v=ch in 'aeiouy'
            if v and not p: c+=1
            p=v
        if w.endswith('e'): c-=1
        return max(1,c)
    def _se(self,items):
        if not items: return 0.0
        return scipy_entropy([c/len(items) for c in Counter(items).values()],base=2)
    def _burst(self,iv):
        if len(iv)<2: return 0.0
        m,s=np.mean(iv),np.std(iv); return (s-m)/(s+m) if (s+m)>0 else 0.0
    def _ng(self,w,n): return [tuple(w[i:i+n]) for i in range(len(w)-n+1)] if len(w)>=n else []
    def _cc(self,x,y):
        if len(x)<2 or np.std(x)==0 or np.std(y)==0: return 0.0
        v=np.corrcoef(x,y)[0,1]; return v if not np.isnan(v) else 0.0
    def _cs(self,v1,v2):
        n1,n2=np.linalg.norm(v1),np.linalg.norm(v2); return np.dot(v1,v2)/(n1*n2) if n1>0 and n2>0 else 0.0
    def _ppl(self,text):
        try:
            self._load_gpt2()
            enc=self.gpt2_tokenizer(text,return_tensors='pt',truncation=True,max_length=1024,padding=True)
            ids=enc['input_ids'].to(self.device)
            if ids.shape[1]<2: return [0.0]*8
            with torch.no_grad():
                out=self.gpt2_model(ids,labels=ids); loss=out.loss.item()
                tl=nn.CrossEntropyLoss(reduction='none')(out.logits[...,:-1,:].contiguous().view(-1,out.logits.size(-1)),ids[...,1:].contiguous().view(-1)).cpu().numpy()
            p25,p75=np.percentile(tl,[25,75]); mid=len(tl)//2
            return [min(math.exp(loss),10000),float(np.mean(tl)),min(float(np.max(tl)),50),float(np.std(tl)),float(stats.skew(tl)) if len(tl)>2 else 0.0,float(np.sum(tl<p25)/len(tl)),float(np.sum(tl>p75)/len(tl)),float(np.mean(tl[mid:])-np.mean(tl[:mid])) if mid>0 else 0.0]
        except: return [0.0]*8
    def extract_single(self,text):
        if not text or not isinstance(text,str): return np.zeros(NUM_RAW_FEATURES,dtype=np.float32)
        text=str(text).strip()
        if len(text)<20: return np.zeros(NUM_RAW_FEATURES,dtype=np.float32)
        w=self._tok(text); s=self._sent(text)
        if len(w)<10: return np.zeros(NUM_RAW_FEATURES,dtype=np.float32)
        f=[]
        f.extend(self._ppl(text))
        chars=[c for c in text.lower() if c.isalnum() or c.isspace()]; bg=self._ng(w,2); tg=self._ng(w,3)
        ew=self._se(w); ebg=self._se(bg)
        shapes=['UPPER' if x.isupper() else 'Title' if x and x[0].isupper() else 'HasNum' if any(c.isdigit() for c in x) else 'lower' for x in w]
        sl=[len(x.split()) for x in s]
        f.extend([self._se(chars),ew,ebg,self._se(tg),max(0,ebg-ew) if ew>0 else 0.0,ew/math.log2(len(set(w))+1) if w else 0.0,self._se(shapes),self._se([c for c in text if c in string.punctuation]) or 0.0,self._se([min(l//5,10) for l in sl]) if sl else 0.0,self._se([min(len(x),15) for x in w])])
        wp=defaultdict(list)
        for i,x in enumerate(w): wp[x].append(i)
        ivs=[p[i]-p[i-1] for p in wp.values() if len(p)>1 for i in range(1,len(p))]
        pp=[i for i,c in enumerate(text) if c in '.,!?;:']
        pi=[pp[i]-pp[i-1] for i in range(1,len(pp))] if len(pp)>1 else []; m=np.mean(sl) if sl else 0
        f.extend([self._burst(ivs) if ivs else 0.0,np.std(sl)/m if m>0 and len(sl)>1 else 0.0,self._burst(pi) if pi else 0.0,self._cc(ivs[:-1],ivs[1:]) if len(ivs)>2 else 0.0,float(np.var(ivs)/np.mean(ivs)) if ivs and np.mean(ivs)>0 else 0.0,float(np.std(ivs)/(np.mean(ivs)+1)) if ivs else 0.0])
        if len(w)<5: f.extend([0.0]*10)
        else:
            def sb(ngl):
                if len(ngl)<2: return 0.0
                sc=[]
                for i2,ref in enumerate(ngl):
                    oth=[item for j,ng in enumerate(ngl) if j!=i2 for item in ng]
                    if not oth or not ref: continue
                    rc,oc=Counter(ref),Counter(oth); sc.append(sum((rc&oc).values())/sum(rc.values()) if sum(rc.values())>0 else 0)
                return float(np.mean(sc)) if sc else 0.0
            sw=[self._tok(x) for x in s if x.strip()]; bc,tc=Counter(bg),Counter(tg)
            rb=sum(1 for c in bc.values() if c>1); rt=sum(1 for c in tc.values() if c>1)
            comp=len(zlib.compress(text.encode('utf-8'),9))/len(text.encode('utf-8')) if text else 1.0
            sn=[' '.join(self._tok(x)) for x in s]
            f.extend([sb([self._ng(x,2) for x in sw if len(x)>=2]),sb([self._ng(x,3) for x in sw if len(x)>=3]),sb([self._ng(x,4) for x in sw if len(x)>=4]),len(set(bg))/len(bg) if bg else 0.0,len(set(tg))/len(tg) if tg else 0.0,rb/len(bc) if bc else 0.0,rt/len(tc) if tc else 0.0,max(bc.values())/len(bg) if bg else 0.0,comp,(len(sn)-len(set(sn)))/len(s) if s else 0.0])
        if len(w)<10: f.extend([0.0]*8)
        else:
            wc=Counter(w); freq=sorted(wc.values(),reverse=True); ranks=np.log(np.arange(1,len(freq)+1)); fl=np.log(np.array(freq)+1)
            zipf=abs(np.polyfit(ranks,fl,1)[0]+1) if len(ranks)>1 else 0.0; M1,M2=len(w),sum(c*c for c in wc.values())
            hapax=sum(1 for c in wc.values() if c==1); dis=sum(1 for c in wc.values() if c==2)
            vs=[len(set(w[:i2])) for i2 in range(100,len(w)+1,100)]
            heaps=np.polyfit(np.log(np.arange(100,len(w)+1,100)),np.log(vs),1)[0] if len(vs)>2 else 0.5
            ttr=len(set(w))/len(w); wn=50
            mattr=float(np.mean([len(set(w[i2:i2+wn]))/wn for i2 in range(0,len(w)-wn+1,wn//2)])) if len(w)>=wn else ttr
            f.extend([zipf,10000*(M2-M1)/(M1*M1) if M1!=M2 else 0.0,hapax/len(w),dis/len(w),heaps,ttr,mattr,len([x for x in w if x not in self.function_words])/len(w)])
        if len(s)<2: f.extend([0.0]*6)
        else:
            vocab=list(set(w)); w2i={x:i2 for i2,x in enumerate(vocab)}
            def sv(sent):
                v=np.zeros(len(vocab))
                for x in self._tok(sent):
                    if x in w2i: v[w2i[x]]+=1
                n=np.linalg.norm(v); return v/n if n>0 else v
            vecs=[sv(x) for x in s]; cs=[self._cs(vecs[i2],vecs[i2+1]) for i2 in range(len(vecs)-1)]
            sm=float(np.mean(cs)) if cs else 0.0; sv2=float(np.var(cs)) if len(cs)>1 else 0.0
            mid2=len(cs)//2; td=abs(np.mean(cs[:mid2])-np.mean(cs[mid2:])) if len(cs)>2 else 0.0
            all_s=[self._cs(vecs[i2],vecs[j]) for i2 in range(len(vecs)) for j in range(i2+1,min(i2+5,len(vecs)))]
            se=[self._se(self._tok(x)) for x in s]
            f.extend([sm,sv2,td,float(np.mean(all_s)) if all_s else 0.0,float(np.mean(se)) if se else 0.0,1.0/(1.0+sv2)])
        if not w or not s: f.extend([0.0]*6)
        else:
            nw,ns=len(w),len(s); nsyl=sum(self._syl(x) for x in w); nc=sum(len(x) for x in w)
            L,S=(nc/nw)*100,(ns/nw)*100; cw=sum(1 for x in w if self._syl(x)>=3)
            f.extend([206.835-1.015*(nw/ns)-84.6*(nsyl/nw),0.39*(nw/ns)+11.8*(nsyl/nw)-15.59,4.71*(nc/nw)+0.5*(nw/ns)-21.43,0.0588*L-0.296*S-15.8,1.0430*math.sqrt(cw*(30/ns))+3.1291 if ns>0 else 0.0,0.1579*(cw/nw*100)+0.0496*(nw/ns)+(3.6365 if cw/nw>0.05 else 0.0)])
        if not w: f.extend([0.0]*8)
        else:
            nw=len(w); sl2=[len(x.split()) for x in s]; ac=sum(1 for c in text if c.isalpha())
            f.extend([sum(1 for x in w if x in self.function_words)/nw,sum(1 for x in w if x in self.pronouns)/nw,sum(1 for x in w if x in self.conjunctions)/nw,float(np.mean([len(x) for x in w])),float(np.mean(sl2)) if sl2 else 0.0,float(np.var(sl2)) if len(sl2)>1 else 0.0,sum(1 for c in text if c in string.punctuation)/len(text) if text else 0.0,sum(1 for c in text if c.isupper())/ac if ac>0 else 0.0])
        return np.array(f,dtype=np.float32)
    def extract_batch(self,texts,desc=""):
        f=[self.extract_single(t) for t in tqdm(texts,desc=f"  {desc}")]
        return np.nan_to_num(np.array(f,dtype=np.float32),nan=0.0,posinf=0.0,neginf=0.0)
    def fit(self,features): self.scaler=RobustScaler(); self.scaler.fit(np.nan_to_num(features,nan=0.0,posinf=0.0,neginf=0.0))
    def transform(self,features): return self.scaler.transform(np.nan_to_num(features,nan=0.0,posinf=0.0,neginf=0.0))


# =============================================================================
# MODEL (FlexibleDetector — identical to v2)
# =============================================================================
