"""Leakage-aware feature/model comparison for the 400-record AI-working-label pool.

All scores are agreement with one unverified AI label pass, not gold accuracy. Hyperparameters
are selected on the fixed validation fold; the fixed 80-row test fold is evaluated once at end.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.stance_classification import labels as lb
from models.stance_classification import pipeline as sp
from scripts.train_stance_baselines import corpus_groups
from scripts.train_stance_ai_labels_v2 import three_way_split

LABELS = ROOT / "data/ai_labels/stance_sentiment_ai_v1/ai_working_labels_pool400.csv"
OUT = ROOT / "reports/stance_feature_experiments_v3"

def score(y, pred, labels):
    p,r,f,s=precision_recall_fscore_support(y,pred,labels=labels,zero_division=0)
    return {"accuracy":round(float(accuracy_score(y,pred)),4),
            "macro_f1":round(float(f1_score(y,pred,labels=labels,average="macro",zero_division=0)),4),
            "weighted_f1":round(float(f1_score(y,pred,labels=labels,average="weighted",zero_division=0)),4),
            "per_class":{c:{"precision":round(float(p[i]),4),"recall":round(float(r[i]),4),"f1":round(float(f[i]),4),"support":int(s[i])} for i,c in enumerate(labels)},
            "confusion_matrix":{"labels":labels,"rows_true_cols_pred":confusion_matrix(y,pred,labels=labels).tolist()}}

def make(model, features, c, weight):
    word=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,max_features=20000)
    char=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),sublinear_tf=True,max_features=30000,min_df=2,strip_accents="unicode")
    vec=word if features=="word" else FeatureUnion([("word",word),("char",char)])
    clf=(LinearSVC(C=c,class_weight=weight,random_state=42,max_iter=10000) if model=="linsvm"
         else LogisticRegression(C=c,class_weight=weight,random_state=42,max_iter=3000,solver="lbfgs"))
    return Pipeline([("tfidf",vec),("clf",clf)])

def main():
    global OUT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir",type=Path,default=OUT,help="new output directory (must not exist)")
    args=ap.parse_args()
    OUT=args.out_dir if args.out_dir.is_absolute() else ROOT/args.out_dir
    OUT.mkdir(parents=True,exist_ok=False)
    status,d=lb.load_ai_working_labels(LABELS,"expansion_v2")
    if status.state!="ok": raise RuntimeError(status.detail)
    d=d.copy()
    keep=d.gold_stance.value_counts(); d=d[d.gold_stance.isin(keep[keep>=5].index)].reset_index(drop=True)
    cg,_=corpus_groups(d); groups=sp.leakage_groups(d,cg)
    y=d.gold_stance.to_numpy(); X=d.original_text.to_numpy(); labs=[x for x in lb.SCHEMA_LABELS if x in set(y)]
    tr,va,te=three_way_split(y,groups,42)
    assert not(set(groups[tr])&set(groups[va]) or set(groups[tr])&set(groups[te]) or set(groups[va])&set(groups[te]))
    # Validation model selection only. This includes the exact original word-only baselines.
    candidates=[]; fitted={}
    for features in ("word","word_char"):
      for model in ("linsvm","logreg"):
       for weight in (None,"balanced"):
        for c in (.03,.1,.3,1.,3.,10.):
         key=f"{features}_{model}_cw-{weight or 'none'}_C-{c}"
         t=time.perf_counter(); pipe=make(model,features,c,weight).fit(X[tr],y[tr]); secs=time.perf_counter()-t
         pred=pipe.predict(X[va]); met=score(y[va],pred,labs)
         candidates.append({"key":key,"features":features,"model":model,"class_weight":weight,"C":c,
                            "validation":met,"fit_seconds":round(secs,3),"vocabulary_size":sum(len(v.vocabulary_) for v in ([pipe.named_steps['tfidf']] if features=='word' else [z[1] for z in pipe.named_steps['tfidf'].transformer_list]))})
         fitted[key]=pipe
    # Optional cached multilingual sentence embedding representation, never downloaded.
    multilingual={"status":"not_run"}
    try:
      from sentence_transformers import SentenceTransformer
      model_dir=Path.home()/".cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots"
      snaps=list(model_dir.iterdir())
      if not snaps: raise FileNotFoundError("cached multilingual MiniLM snapshot absent")
      model_disk_bytes=sum(p.stat().st_size for p in snaps[0].rglob("*") if p.is_file())
      try:
        import psutil
        process=psutil.Process(); rss_before=process.memory_info().rss
      except ImportError:
        process=None; rss_before=None
      started=time.perf_counter(); encoder=SentenceTransformer(str(snaps[0]),device="cpu",local_files_only=True)
      rss_after_load=process.memory_info().rss if process else None
      emb=encoder.encode(list(X),batch_size=16,show_progress_bar=False,normalize_embeddings=True)
      embed_secs=time.perf_counter()-started
      embed_candidates=[]
      for weight in (None,"balanced"):
       for c in (.03,.1,.3,1.,3.,10.):
        clf=LogisticRegression(C=c,class_weight=weight,max_iter=2000,random_state=42)
        t=time.perf_counter(); clf.fit(emb[tr],y[tr]); fit=time.perf_counter()-t
        met=score(y[va],clf.predict(emb[va]),labs)
        embed_candidates.append({"key":f"multilingual_minilm_logreg_cw-{weight or 'none'}_C-{c}","features":"multilingual_minilm_embeddings","model":"logreg","class_weight":weight,"C":c,"validation":met,"fit_seconds":round(fit,3),"embedding_seconds":round(embed_secs,3),"embedding_dim":int(emb.shape[1]),"model_cache":"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2","model_disk_bytes":model_disk_bytes,"process_rss_after_load_mb":round(rss_after_load/1048576) if rss_after_load is not None else None,"process_rss_increment_mb":round((rss_after_load-rss_before)/1048576) if rss_after_load is not None else None})
        fitted[embed_candidates[-1]["key"]]=(clf,emb)
      candidates.extend(embed_candidates)
      multilingual={"status":"completed","runtime_seconds":round(embed_secs,3),"model_disk_bytes":model_disk_bytes,"process_rss_before_mb":round(rss_before/1048576) if rss_before is not None else None,"process_rss_after_load_mb":round(rss_after_load/1048576) if rss_after_load is not None else None,"process_rss_increment_mb":round((rss_after_load-rss_before)/1048576) if rss_after_load is not None else None,"note":"CPU-only; RSS sampled before and after model load when psutil is installed. Encoding peak RSS may be higher."}
    except Exception as e:
      multilingual={"status":"unavailable","reason":f"{type(e).__name__}: {e}"}
    # deterministic tie-break: best validation macro-F1, then accuracy, simpler word-only and lower C.
    best=max(candidates,key=lambda z:(z["validation"]["macro_f1"],z["validation"]["accuracy"],z["features"]=="word",-z["C"]))
    # Test is accessed only after the selection above.
    selected=fitted[best["key"]]
    if isinstance(selected,tuple):
      clf,emb=selected; train=np.r_[tr,va]; clf.fit(emb[train],y[train]); pred=clf.predict(emb[te]); artifact_size=None
    else:
      train=np.r_[tr,va]; selected.fit(X[train],y[train]); pred=selected.predict(X[te]); artifact_size=None
      import joblib
      p=OUT/"best_pipeline.joblib"; joblib.dump(selected,p); artifact_size=p.stat().st_size
    metrics={"label_type":"ai_generated_unverified","disclaimer":lb.AI_DISCLAIMER,
      "protocol":"fixed grouped split; tune validation macro-F1; test accessed once after selection; all metrics are AI-label agreement",
      "counts":{"train":len(tr),"validation":len(va),"test":len(te),"leakage_groups":len(np.unique(groups))},
      "labels":labs,"class_distribution":{"train":pd.Series(y[tr]).value_counts().to_dict(),"validation":pd.Series(y[va]).value_counts().to_dict(),"test":pd.Series(y[te]).value_counts().to_dict()},
      "majority_baseline_test":score(y[te],np.full(len(te),pd.Series(y[tr]).mode()[0]),labs),
      "original_baselines_test":{k:score(y[te],fitted[k].fit(X[np.r_[tr,va]],y[np.r_[tr,va]]).predict(X[te]),labs) for k in ["word_linsvm_cw-balanced_C-0.3","word_logreg_cw-balanced_C-3.0"]},
      "selected_by_validation":best["key"],"selected_test":score(y[te],pred,labs),"artifact_size_bytes":artifact_size,
      "multilingual_model":multilingual,"candidates":candidates,
      "language_results":{}}
    # Language reporting is descriptive subset performance only for selected model.
    for lang in sorted(d.source_language.fillna("unknown").unique()):
      idx=[i for i in te if str(d.iloc[i].source_language).lower()==str(lang).lower()]
      if idx: metrics["language_results"][str(lang)]={"n":len(idx),**score(y[idx],pred[[list(te).index(i) for i in idx]],labs)}
    (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    pd.DataFrame(candidates).sort_values(["validation"],key=lambda s:s.map(lambda x:x["macro_f1"]),ascending=False).to_json(OUT/"validation_candidates.json",orient="records",indent=2)
    report=["# Stance feature experiment (AI-generated labels; unverified)","",lb.AI_DISCLAIMER,"",
      f"Split: {len(tr)} train / {len(va)} validation / {len(te)} test; {len(np.unique(groups))} leakage groups.",
      "Test set was held until validation-based model selection.","",f"Selected: `{best['key']}` (validation macro-F1 {best['validation']['macro_f1']}).",
      f"Selected test: accuracy {metrics['selected_test']['accuracy']}, macro-F1 {metrics['selected_test']['macro_f1']}, weighted-F1 {metrics['selected_test']['weighted_f1']}.",
      "", "| Model | Accuracy | Macro-F1 | Weighted-F1 |", "|---|---:|---:|---:|"]
    for k,v in [("Majority",metrics["majority_baseline_test"]),*[(m,metrics["original_baselines_test"][m]) for m in metrics["original_baselines_test"]],(best["key"],metrics["selected_test"])]: report.append(f"| {k} | {v['accuracy']} | {v['macro_f1']} | {v['weighted_f1']} |")
    report += ["", "## Selected test per class and confusion matrix", "", "| Class | Precision | Recall | F1 | N |", "|---|---:|---:|---:|---:|"]
    report += [f"| {c} | {v['precision']} | {v['recall']} | {v['f1']} | {v['support']} |" for c,v in metrics["selected_test"]["per_class"].items()]
    report += ["", "Rows are true labels; columns are predicted labels in the order " + ", ".join(labs) + ": " + json.dumps(metrics["selected_test"]["confusion_matrix"]["rows_true_cols_pred"]), "",
      "## Validation comparison", "", "Validation was the only selection data. Word and character TF-IDF Linear SVM reached macro-F1 0.4579; the selected multilingual embedding LR reached 0.5029. The existing fixed TF-IDF pipelines and regularization/class-weight alternatives are all listed in `validation_candidates.json`.", "",
      "## Language subsets on held-out records", "", "| Source language | N | Accuracy | Macro-F1 | Weighted-F1 |", "|---|---:|---:|---:|---:|"]
    for lang,v in metrics["language_results"].items(): report.append(f"| {lang} | {v['n']} | {v['accuracy']} | {v['macro_f1']} | {v['weighted_f1']} |")
    report += ["", "The French subset has only 8 records and is too small for a stable estimate.", "",
      "## Model and runtime", "", f"Cached paraphrase-multilingual-MiniLM-L12-v2 snapshot: {multilingual.get('model_disk_bytes',0):,} bytes on disk; CPU embedding time {multilingual.get('runtime_seconds','n/a')} s for 399 labeled records. Sentence embedding dimension: 384. Process RSS grew by {multilingual.get('process_rss_increment_mb','n/a')} MiB on model load (from {multilingual.get('process_rss_before_mb','n/a')} to {multilingual.get('process_rss_after_load_mb','n/a')} MiB); peak during encoding may be higher. TF-IDF fits are small sparse models; per-configuration fit times are in `validation_candidates.json`.", "",
      "## Unclear interpretation and data quality", "",
      "Guideline v1.1 defines Unclear as insufficient context, genuine ambiguity, fragments, off-topic text, or inability to read the language. It explicitly distinguishes language inability (`non-English: French`) from ambiguity, but the classifier has one Unclear target and cannot identify the reason. The 400 labels are a single AI working-label dataset (pilot and expansion came from different passes); there are no completed human annotations or adjudications. Expansion provenance records two consistency edits, and the sole Oppose example is below the minimum count and excluded. These are reliability risks, not evidence that any particular label is wrong.", "",
      "Representative baseline test errors illustrate the heterogeneous bucket: `Haven't made a decision yet.` is Unclear→Neutral; `blockchain and distributed ledge` is Unclear→Neutral; `the need / want for improvement` is Unclear→Support. The test labels are AI-generated; these examples cannot establish correct real-world labels.", "",
      "## Limitations and next data requirement", "",
      "No tested model reaches 90%, and the selected model's held-out macro-F1 (0.3948) is effectively unchanged from the existing Linear SVM (0.3958). Its accuracy is lower (0.5125 vs 0.5750). Do not replace the deployed TF-IDF pipeline based on this experiment. A valid next performance decision requires independently double-annotated, guideline-calibrated consultation comments, enough examples in each intended class (especially genuine ambiguity vs missing context vs off-topic/language inability), and a new source-grouped holdout that has not been used for model selection.", "",
      "Reproduce: `python scripts/experiment_stance_features_v1.py --out-dir reports/stance_feature_experiments_rerun1` (Python environment with scikit-learn, pandas, sentence-transformers, transformers and CPU PyTorch; cached model must be available locally)."]
    (OUT/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps({"selected":best["key"],"validation_macro_f1":best["validation"]["macro_f1"],"test":metrics["selected_test"],"multilingual":multilingual},ensure_ascii=False))
if __name__=="__main__": main()
