"""Full 5,000 COCO Karpathy Image Captioning Evaluation with Calibrated Prompting.
Features:
  - Instant SGG cache loading (5,000 precomputed Scene Graphs in artifacts/coco_5000_sgg_native_cache.json)
  - Calibrated prompt architecture: 9-13 words, negative constraint, action-grounded demonstrations
  - Batched Qwen2.5-14B inference (B=8)
  - Comprehensive pycocoevalcap evaluation on all 5,000 images vs 25,010 GT references.
"""
import os, sys, json, io, time, gc, argparse
from pathlib import Path
import torch
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['native', 'feedback'], default='native')
    parser.add_argument('--prompt', choices=['calibrated', 'few', 'zero'], default='calibrated')
    parser.add_argument('--batch-size', type=int, default=8)
    return parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

java_bin = os.path.expanduser('~/rsg-former/.tools/java/temurin8-52dcd578baca/jdk8u504-b01-jre/bin')
if os.path.exists(java_bin):
    os.environ['PATH'] = java_bin + ':' + os.environ['PATH']

ROOT = Path.home() / 'rsg-former'
SHARED = Path.home() / 'shared-assets/datasets'
sys.path.insert(0, str(ROOT / '.tools/deps'))
sys.path.insert(0, str(ROOT / 'src'))

from rsgformer.vg_v2_runtime import native_model, capture, Adaptation
from rsgformer.vg_native_postprocess import native_vg_postprocess
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

def main():
    args = parse_args() if len(sys.argv) > 1 else argparse.Namespace(variant='native', prompt='calibrated', batch_size=8)
    
    START_TIME = time.time()
    print(f"[{time.strftime('%X')}] Starting Full 5,000 COCO Karpathy Captioning Pipeline...")
    print(f"  Configuration: variant={args.variant}, prompt={args.prompt}, batch_size={args.batch_size}")

    manifest_path = SHARED / 'coco_karpathy/test_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    test_images = manifest['images']
    print(f"Loaded {len(test_images)} test images from manifest.")

    meta = json.loads((ROOT / 'data/vg150/VG-SGG-dicts-with-attri.json').read_text())
    obj_names = meta['idx_to_label']
    pred_names = meta['idx_to_predicate']

    def extract_triplets_and_objects(pred, eq):
        triplets = []
        seen_triples = set()
        detected_objects = set()
        
        for q_idx in range(len(pred['pred_boxes'])):
            cls_id = int(pred['pred_classes'][q_idx]) + 1
            detected_objects.add(obj_names.get(str(cls_id), f'obj_{cls_id}'))
            
        for rank, ((s, o), scores, score) in enumerate(zip(pred['pred_rel_inds'], pred['rel_scores'], pred['ranking_scores']), 1):
            sid, oid = int(eq[s]), int(eq[o])
            label = int(scores.argmax()) + 1
            if sid == oid or score <= 0: continue
            s_cls = int(pred['pred_classes'][s]) + 1
            o_cls = int(pred['pred_classes'][o]) + 1
            s_name = obj_names.get(str(s_cls), f'obj_{s_cls}')
            o_name = obj_names.get(str(o_cls), f'obj_{o_cls}')
            p_name = pred_names.get(str(label), f'pred_{label}')
            
            triple_key = (s_name, p_name, o_name)
            if triple_key in seen_triples: continue
            seen_triples.add(triple_key)
            triplets.append({'subject': s_name, 'predicate': p_name, 'object': o_name, 'score': float(score)})
            if len(triplets) >= 8: break
            
        return triplets, sorted(detected_objects)[:10]

    # --- STAGE 1: SGG EXTRACTION ---
    cache_sgg_path = ROOT / f'artifacts/coco_5000_sgg_{args.variant}_cache.json'
    sgg_cache = {}
    if cache_sgg_path.exists():
        try:
            sgg_cache = json.loads(cache_sgg_path.read_text())
            print(f"Loaded existing SGG cache: {len(sgg_cache)} extractions ready.")
        except Exception as e:
            print(f"Warning: could not parse existing SGG cache ({e}), starting fresh.")
            sgg_cache = {}

    images_to_extract = [entry for entry in test_images if str(entry['id']) not in sgg_cache]
    if len(images_to_extract) > 0:
        print(f"[{time.strftime('%X')}] Extracting remaining {len(images_to_extract)} images with SGG...")
        base_model, _, _ = native_model(ROOT)
        ckpt_path = ROOT / 'artifacts/vg_method_v2_speed_m2_v1/confirmation_native_1729/best.pt'
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        base_model.load_state_dict(ck['native_head'], strict=False)
        adapt = Adaptation(base_model, args.variant).cuda().eval()
        base_model.eval()

        sgg_start = time.time()
        for idx, entry in enumerate(images_to_extract, 1):
            img_id_str = str(entry['id'])
            img_path = SHARED / 'coco_karpathy/images' / entry['file_name']
            image = Image.open(img_path).convert('RGB')
            with torch.no_grad():
                item, full = capture(base_model, image)
                batch = {k: item[k][None].float().cuda() for k in ['nodes', 'boxes', 'logits', 'visual']}
                out, _ = adapt(batch)
                pred = native_vg_postprocess(out, *image.size)
            triplets, objs = extract_triplets_and_objects(pred, pred['query_to_nms_equivalence_id'])
            sgg_cache[img_id_str] = {'triplets': triplets, 'objects': objs, 'file_name': entry['file_name']}
            
            if idx % 250 == 0 or idx == len(images_to_extract):
                cache_sgg_path.parent.mkdir(parents=True, exist_ok=True)
                cache_sgg_path.write_text(json.dumps(sgg_cache))

        del base_model, adapt, ck, item, full, batch, out, pred
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"[{time.strftime('%X')}] All 5,000 SGG scene graphs are already precomputed!")

    # --- STAGE 2: CAPTION GENERATION ---
    captions_cache_path = ROOT / f'artifacts/coco_5000_captions_{args.variant}_{args.prompt}_cache.json'
    captions_cache = {}
    if captions_cache_path.exists():
        try:
            captions_cache = json.loads(captions_cache_path.read_text())
            print(f"Loaded existing captions cache: {len(captions_cache)} captions ready.")
        except Exception:
            captions_cache = {}

    images_to_caption = [entry for entry in test_images if str(entry['id']) not in captions_cache]
    print(f"Images to generate captions: {len(images_to_caption)}")

    if len(images_to_caption) > 0:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_path = '/home/stackops/etechs/msvd-tkg-captioning/models/qwen2.5-14b'
        print(f"[{time.strftime('%X')}] Loading Qwen-14B...")
        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
        tok.padding_side = 'left'
        
        qwen = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            local_files_only=True
        )
        qwen.eval()
        print(f"[{time.strftime('%X')}] Qwen-14B loaded.")

        if args.prompt == 'calibrated':
            sys_content = (
                "You are an expert image captioner. You are given detected visual relations from an image. "
                "Write exactly ONE natural English sentence (9 to 13 words) describing the main activity and setting. "
                "Do not describe clothing or body parts if an action or interaction is present. "
                "Do not speculate on weather, mood, or unseen items. Be direct, factual, and fluent."
            )
            demos = (
                "EXAMPLE 1:\n"
                "Detected visual relations: man riding motorcycle; motorcycle on dirt road; trees near road\n"
                "Description: A man is riding a motorcycle down a dirt road near trees.\n\n"
                "EXAMPLE 2:\n"
                "Detected visual relations: woman sitting at table; bowl of soup on table; woman holding fork\n"
                "Description: A woman is sitting at a wooden table eating from a bowl.\n\n"
                "EXAMPLE 3:\n"
                "Detected visual relations: dog chasing ball; dog running on grass; ball in grass\n"
                "Description: A dog is running across the green grass chasing after a ball.\n\n"
            )
            max_new = 28
        else:
            sys_content = (
                "You are an expert image captioner. You are given visual evidence detected in an image. "
                "Write exactly ONE concise, natural English sentence (8 to 15 words) describing the primary activity or scene. "
                "Do not include object IDs, lists, bullet points, or metadata. Be factual and fluent."
            )
            demos = (
                "EXAMPLE 1:\n"
                "Detected visual relations: dog playing with ball; dog running on grass; ball in grass\n"
                "Description: A brown dog running across the green grass with a ball in its mouth.\n\n"
                "EXAMPLE 2:\n"
                "Detected visual relations: person holding umbrella; person walking on sidewalk; car on street\n"
                "Description: A person walking down a city sidewalk under an umbrella in the rain.\n\n"
            )
            max_new = 32

        def format_triplets(triplets, fallback_objs):
            if triplets:
                return '; '.join([f"{t['subject']} {t['predicate']} {t['object']}" for t in triplets[:8]])
            return ', '.join(fallback_objs[:5])

        BATCH_SIZE = args.batch_size
        gen_start = time.time()
        
        for b_idx in range(0, len(images_to_caption), BATCH_SIZE):
            batch_entries = images_to_caption[b_idx:b_idx + BATCH_SIZE]
            prompts = []
            for entry in batch_entries:
                info = sgg_cache[str(entry['id'])]
                rel_str = format_triplets(info['triplets'], info['objects'])
                user_msg = demos + f"Detected visual relations: {rel_str}\nDescription:"
                msg = [{'role': 'system', 'content': sys_content}, {'role': 'user', 'content': user_msg}]
                prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
                
            inputs = tok(prompts, return_tensors='pt', padding=True).to('cuda')
            prompt_len = inputs.input_ids.shape[1]
            
            with torch.no_grad():
                outputs = qwen.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id
                )
                
            for i, entry in enumerate(batch_entries):
                gen_tokens = outputs[i][prompt_len:]
                gen_text = tok.decode(gen_tokens, skip_special_tokens=True).strip().split('\n')[0].strip()
                captions_cache[str(entry['id'])] = gen_text
                
            done_count = b_idx + len(batch_entries)
            if done_count % 200 == 0 or done_count == len(images_to_caption):
                elapsed = time.time() - gen_start
                rate = done_count / elapsed
                remain = (len(images_to_caption) - done_count) / max(rate, 0.01)
                print(f"  [Qwen] {done_count}/{len(images_to_caption)} ({done_count/len(images_to_caption)*100:.1f}%) | "
                      f"Speed: {rate:.2f} img/s | ETA: {remain/60:.1f}m", flush=True)
                      
            if done_count % 250 == 0 or done_count == len(images_to_caption):
                captions_cache_path.write_text(json.dumps(captions_cache))

        del qwen, tok
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{time.strftime('%X')}] Caption generation completed. Cache saved. VRAM freed.")
    else:
        print(f"[{time.strftime('%X')}] All 5,000 captions already in cache.")

    # --- STAGE 3: FULL EVALUATION ---
    print(f"[{time.strftime('%X')}] Evaluating all 5,000 images against 25,010 ground truth references...")
    ref_path = SHARED / 'coco_karpathy/test_references_coco.json'
    refs_data = json.loads(ref_path.read_text())

    gts_raw = {}
    for ann in refs_data['annotations']:
        img_id = ann['image_id']
        if str(img_id) in captions_cache:
            gts_raw.setdefault(img_id, []).append({'caption': ann['caption']})

    preds_raw = {int(k): [{'caption': v}] for k, v in captions_cache.items()}

    tokenizer = PTBTokenizer()
    print("Tokenizing references...")
    gts_tok = tokenizer.tokenize(gts_raw)
    print("Tokenizing predictions...")
    res_tok = tokenizer.tokenize(preds_raw)

    scores = {}
    print("Computing BLEU scores (1-4)...")
    b_score, _ = Bleu(4).compute_score(gts_tok, res_tok)
    for i, s in enumerate(b_score, 1):
        scores[f'Bleu_{i}'] = float(s) * 100.0

    print("Computing METEOR score...")
    meteor_scorer = Meteor()
    try:
        m_score, _ = meteor_scorer.compute_score(gts_tok, res_tok)
        scores['METEOR'] = float(m_score) * 100.0
    finally:
        if hasattr(meteor_scorer, 'lock') and meteor_scorer.lock.locked():
            meteor_scorer.lock.release()
        del meteor_scorer

    print("Computing ROUGE-L score...")
    r_score, _ = Rouge().compute_score(gts_tok, res_tok)
    scores['ROUGE_L'] = float(r_score) * 100.0

    print("Computing CIDEr score...")
    c_score, _ = Cider().compute_score(gts_tok, res_tok)
    scores['CIDEr'] = float(c_score) * 100.0

    total_wall = time.time() - START_TIME
    print(f"[{time.strftime('%X')}] Evaluation complete in {total_wall:.1f}s!")
    print("\n" + "="*60)
    print(f"FULL 5,000 COCO KARPATHY BENCHMARK RESULTS ({args.variant.upper()} + {args.prompt.upper()}):")
    print(f"  CIDEr   : {scores['CIDEr']:.2f}")
    print(f"  BLEU-4  : {scores['Bleu_4']:.2f}")
    print(f"  BLEU-1  : {scores['Bleu_1']:.2f}")
    print(f"  METEOR  : {scores['METEOR']:.2f}")
    print(f"  ROUGE-L : {scores['ROUGE_L']:.2f}")
    print("="*60 + "\n")

    report = {
        'status': 'COCO_FULL5000_EVALUATION_COMPLETE',
        'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'variant': args.variant,
        'prompt_strategy': args.prompt,
        'batch_size': args.batch_size,
        'total_images': len(captions_cache),
        'wall_seconds': total_wall,
        'metrics': scores,
        'qualitative_samples': [
            {
                'image_id': entry['id'],
                'file_name': entry['file_name'],
                'generated_caption': captions_cache.get(str(entry['id'])),
                'triplets': sgg_cache.get(str(entry['id']), {}).get('triplets', []),
                'ground_truth': [a['caption'].strip() for a in gts_raw.get(entry['id'], [])]
            }
            for entry in test_images[:25]
        ]
    }

    out_path = ROOT / f'reports/COCO_FULL5000_CAPTION_{args.variant.upper()}_{args.prompt.upper()}_RESULT.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Full report saved to {out_path}")

    print("SUMMARY_JSON_START")
    print(json.dumps({'metrics': scores, 'total_images': len(captions_cache), 'wall_seconds': round(total_wall, 1)}, indent=2))
    print("SUMMARY_JSON_END")

if __name__ == '__main__':
    main()
