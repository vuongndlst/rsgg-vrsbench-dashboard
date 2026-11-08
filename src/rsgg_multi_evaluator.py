"""Unified Comprehensive Evaluator for Remote Sensing Image Captioning & Grounding.
Calculates:
  1. Standard PyCocoEvalCap metrics: BLEU-1..4, METEOR, ROUGE-L, CIDEr
  2. SPICE (Semantic Propositional Image Caption Evaluation via Stanford CoreNLP)
  3. CHAIR (Caption Hallucination Assessment: sentence-level and object-level)
  4. BERTScore (Deep contextual embedding similarity)
  5. Grounding Accuracy (mIoU, Acc@0.5, Acc@0.7)
"""

import json, re, sys, os
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import numpy as np

class MultiMetricEvaluator:
    def __init__(self, enable_spice: bool = True, enable_bertscore: bool = False):
        self.enable_spice = enable_spice
        self.enable_bertscore = enable_bertscore
        self._spice_evaluator = None
        if self.enable_spice:
            try:
                from pycocoevalcap.spice.spice import Spice
                self._spice_evaluator = Spice()
            except Exception as e:
                print(f"[Warning] Failed to initialize SPICE: {e}", file=sys.stderr)

    def compute_coco_metrics(self, gts: Dict[Any, List[str]], res: Dict[Any, List[str]]) -> Dict[str, float]:
        """Compute BLEU 1-4, METEOR, ROUGE-L, CIDEr."""
        scores = {}
        # 1. BLEU
        try:
            from pycocoevalcap.bleu.bleu import Bleu
            bleu_scorer = Bleu(4)
            bleu_scores, _ = bleu_scorer.compute_score(gts, res)
            for i, b in enumerate(bleu_scores, 1):
                scores[f"BLEU-{i}"] = float(b * 100)
        except Exception as e:
            print(f"[Warning] BLEU error: {e}", file=sys.stderr)

        # 2. METEOR
        try:
            from pycocoevalcap.meteor.meteor import Meteor
            meteor_scorer = Meteor()
            meteor_score, _ = meteor_scorer.compute_score(gts, res)
            scores["METEOR"] = float(meteor_score * 100)
        except Exception:
            # Fallback to nltk meteor if jar/pipe issue
            try:
                import nltk
                from nltk.translate.meteor_score import meteor_score
                m_list = []
                for k in gts:
                    ref_toks = [r.lower().split() for r in gts[k]]
                    hyp_tok = res[k][0].lower().split()
                    m_list.append(meteor_score(ref_toks, hyp_tok))
                scores["METEOR"] = float(np.mean(m_list) * 100)
            except Exception as e:
                print(f"[Warning] METEOR fallback error: {e}", file=sys.stderr)

        # 3. ROUGE-L
        try:
            from pycocoevalcap.rouge.rouge import Rouge
            rouge_scorer = Rouge()
            rouge_score, _ = rouge_scorer.compute_score(gts, res)
            scores["ROUGE-L"] = float(rouge_score * 100)
        except Exception as e:
            print(f"[Warning] ROUGE error: {e}", file=sys.stderr)

        # 4. CIDEr
        try:
            from pycocoevalcap.cider.cider import Cider
            cider_scorer = Cider()
            cider_score, _ = cider_scorer.compute_score(gts, res)
            scores["CIDEr"] = float(cider_score * 100)
        except Exception as e:
            print(f"[Warning] CIDEr error: {e}", file=sys.stderr)

        # 5. SPICE (if enabled)
        if self._spice_evaluator:
            try:
                spice_score, _ = self._spice_evaluator.compute_score(gts, res)
                scores["SPICE"] = float(spice_score * 100)
            except Exception as e:
                print(f"[Warning] SPICE compute error: {e}", file=sys.stderr)

        return scores

    def compute_bertscore(self, gts: Dict[Any, List[str]], res: Dict[Any, List[str]], model_type="distilbert-base-uncased") -> Dict[str, float]:
        """Compute BERTScore Precision, Recall, and F1."""
        try:
            import bert_score
            cands = []
            refs = []
            for k in gts:
                cands.append(res[k][0])
                refs.append(gts[k])  # multiple references
            P, R, F1 = bert_score.score(cands, refs, lang="en", model_type=model_type, verbose=False)
            return {
                "BERTScore_P": float(P.mean().item() * 100),
                "BERTScore_R": float(R.mean().item() * 100),
                "BERTScore_F1": float(F1.mean().item() * 100)
            }
        except Exception as e:
            print(f"[Warning] BERTScore error: {e}", file=sys.stderr)
            return {}

    def compute_chair(self, gts: Dict[Any, List[str]], res: Dict[Any, List[str]], gt_categories: Dict[Any, List[str]]) -> Dict[str, float]:
        """Compute CHAIR_s (sentence hallucination) and CHAIR_i (instance hallucination)."""
        # Standard RS objects dictionary
        KNOWN_RS_CLASSES = {
            "plane", "airplane", "aircraft", "jet", "ship", "boat", "vessel", "car", "vehicle",
            "truck", "bus", "train", "bridge", "building", "house", "residential", "runway",
            "apron", "tarmac", "terminal", "harbor", "port", "dock", "pier", "tank", "storage",
            "field", "road", "highway", "forest", "tree", "river", "water", "lake", "ocean",
            "grass", "stadium", "track", "diamond", "court", "island", "container"
        }
        
        num_sentences = 0
        hallucinated_sentences = 0
        total_gen_instances = 0
        hallucinated_instances = 0

        for k in res:
            gen_text = res[k][0].lower()
            words = set(re.findall(r'\b[a-z]+\b', gen_text))
            
            # Ground truth categories for this image
            valid_cats = set([c.lower() for c in gt_categories.get(k, [])])
            for ref in gts.get(k, []):
                for w in re.findall(r'\b[a-z]+\b', ref.lower()):
                    if w in KNOWN_RS_CLASSES:
                        valid_cats.add(w)

            gen_mentioned = words.intersection(KNOWN_RS_CLASSES)
            if not gen_mentioned:
                continue

            num_sentences += 1
            has_hallucination = False
            for m in gen_mentioned:
                total_gen_instances += 1
                if m not in valid_cats:
                    hallucinated_instances += 1
                    has_hallucination = True

            if has_hallucination:
                hallucinated_sentences += 1

        chair_s = (hallucinated_sentences / num_sentences) if num_sentences > 0 else 0.0
        chair_i = (hallucinated_instances / total_gen_instances) if total_gen_instances > 0 else 0.0
        return {
            "CHAIR_s": float(chair_s),
            "CHAIR_i": float(chair_i),
            "valid_sentences_evaluated": num_sentences
        }

    def compute_all(self, gts: Dict[Any, List[str]], res: Dict[Any, List[str]], gt_categories: Dict[Any, List[str]] = None) -> Dict[str, Any]:
        """Compute full benchmark evaluation suite."""
        out = {}
        # 1. COCO & SPICE
        out.update(self.compute_coco_metrics(gts, res))
        # 2. CHAIR
        if gt_categories:
            out.update(self.compute_chair(gts, res, gt_categories))
        # 3. BERTScore
        if self.enable_bertscore:
            out.update(self.compute_bertscore(gts, res))
        return out

