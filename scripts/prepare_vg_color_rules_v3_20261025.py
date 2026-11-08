"""Build explicit150-class research rules from TRAIN frequencies, no TEST choices."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json
ROOT=Path(__file__).absolute().parents[1]
audit=ROOT/'reports/VG150_COLOR_PRIORITY_20261024.csv'
rows=list(csv.DictReader(audit.open(encoding='utf-8')));assert len(rows)==150
groups={
 'person_or_skin':set('arm boy child ear eye face finger girl guy hand head kid lady leg man men mouth neck nose people person player skier woman'.split()),
 'transparent_or_emissive':set('bottle cup glass lamp laptop light screen window windshield'.split()),
 'text_symbol':set('letter logo number sign'.split()),
 'container_contents':set('basket bowl box plate pot sink vase'.split()),
 'vehicle_body':set('airplane bike boat bus car engine motorcycle plane train truck vehicle wing'.split()),
 'animal_coat':set('animal bear bird cat cow dog elephant giraffe horse paw sheep tail zebra'.split()),
 'garment_or_accessory':set('bag boot cap coat glove hat helmet jacket jean pant shirt shoe short sneaker sock tie umbrella'.split()),
 'hair':{'hair'},
 'natural_or_food_surface':set('banana beach branch flower food fruit hill leaf mountain orange pizza plant rock snow tree trunk vegetable wave'.split())}
exclusions={
 'person_or_skin':'Never attach clothing/hair color to a whole person or skin-part node. Use its separate garment/hair node; no inferred skin/race labels.',
 'transparent_or_emissive':'Apparent color may belong to contents/reflection/light. Defer unless an opaque surface-specific mask and separate validation exist.',
 'text_symbol':'Do not use CLIP text/OCR response to establish surface color. Defer glyph/color field separation to a later version.',
 'container_contents':'Exclude contained food/objects using independently predicted child masks. Insufficient exposed container pixels -> unknown.',
 'vehicle_body':'Measure visible body/paint; exclude independently predicted windows, windshield, tires/wheels, lights and occluders. A missing reliable body mask -> unknown.',
 'animal_coat':'Use visible coat/feathers only; exclude tack/collar/occluder masks. Report up to two stable substantial colors, not breed-typical color.',
 'garment_or_accessory':'Use the garment/accessory mask, excluding skin and independent occluders. Keep stable main color separate from a smaller stripe/logo.',
 'hair':'Hair-mask only. Blond is a future independently calibrated hair-only term; do not force the basic11 table to invent blond.',
 'natural_or_food_surface':'Visible surface only, no canonical class-color guesses or scene-wide white-balance from object priors.',
 'opaque_surface':'Visible target surface only, exclude independent occluders and check mask/color stability.'}
palette='black blue brown gray green orange pink purple red white yellow'.split()
perclass={};csvrows=[]
for r in rows:
 name=r['class_name'];matches=[k for k,v in groups.items() if name in v];assert len(matches)<=1,(name,matches)
 group=matches[0] if matches else 'opaque_surface'
 disabled=group in ['person_or_skin','transparent_or_emissive','text_symbol']
 support=int(r['train_objects_with_color']);rate=float(r['train_color_percent'])
 priority='not_requested' if disabled else ('high' if support>=100 and rate>=30 else 'secondary')
 item={'class_id':int(r['class_id']),'name':name,'surface_group':group,'request_priority':priority,'production_color_enabled':False,
  'TRAIN_objects':int(r['train_objects']),'TRAIN_color_annotated':support,'TRAIN_color_annotation_percent':rate,
  'rule':exclusions[group],'class_prior_can_choose_color':False,'requires_independent_visual_validation':True}
 perclass[name]=item;csvrows.append(item)
rules={'version':'vg_object_color_rules_v3_20261025','created_utc':datetime.now(timezone.utc).isoformat(),'status':'RESEARCH_RULES_NOT_VALIDATED_PRODUCTION',
 'human_priority':'25Oct research and improve correctness before caption ablations','TRAIN_audit_sha256':hashlib.sha256(audit.read_bytes()).hexdigest(),
 'target_quantity':'visible apparent target-surface color in the source image, not intrinsic reflectance recovered under canonical light',
 'palette':palette,'aliases':{'grey':'gray'},'extended_colors_emitted':False,
 'extended_color_policy':'tan/beige/cream/khaki/maroon require separate color calibration; silver/gold material/appearance and blond hair-only remain separate, not silently normalized in original GT',
 'sampling':'same frozen200 TRAIN target nodes for engineering pilot; no TEST reads, old color-check already observed and cannot be relabeled fresh check',
 'mask_variants':['GrabCut+ColorNames_LUT','SAM2.1_small_bbox+ColorNames_LUT','SAM2.1_small_bbox+ColorNames_LUT+CLIP_contradiction_flag'],
 'finite_operating_points':[
  {'id':'balanced','mask_jitter_iou_min':.75,'primary_mass_min':.60,'color_margin_min':.03,'second_mass_min':.20,'top2_mass_min':.80},
  {'id':'conservative','mask_jitter_iou_min':.85,'primary_mass_min':.70,'color_margin_min':.05,'second_mass_min':.25,'top2_mass_min':.85}],
 'default_operating_point_before_dev_selection':'conservative','maximum_colors':2,'secondary_color_is_primary':False,
 'mask_quality':{'minimum_original_bbox_side_px':32,'minimum_surviving_pixels':256,'bbox_visible_fraction_min':.90,
  'largest_component_share_min':.75,'foreground_fraction_min':.05,'foreground_fraction_max':1.0,'mask_outside_bbox_fraction_max':.05,
  'SAM_predicted_quality_screen_min':.80,'SAM_predicted_quality_is_calibrated_probability':False,'jitter_bbox_fraction':.03,
  'jitter_masks':3,'inner_boundary_erosion_fraction':.01,'erosion_radius_max_px':3,'maximum_parent_child_masks':3},
 'photometry':{'RGB_order':'RGB input explicitly; BGR only in OpenCV GrabCut conversion','lookup':'author32768x11 sRGB table, original R+32G+1024B indexing',
  'image_manipulation_for_statistics':'none; only original mask pixels count; no gray/black background fill, no per-object histogram equalization or gray-world correction',
  'illumination_guard':'compare color on original and mild exposure perturbations (.9,1.1); if primary is unstable, unknown. This is sensitivity, not recovered illumination.',
  'achromatic_policy':'do not discard low-saturation pixels; gray/white/black can be genuine target colors; darkness alone never overrides learned naming',
  'highlight_policy':'report clipping share; refuse severely clipped/underexposed crops, do not delete all white/black pixels',
  'severe_channel_clip_fraction':.60},
 'CLIP':{'role':'secondary contradiction signal, never sole color source or probability','object_prior_selects_color':False,
  'achromatic_can_bypass_CLIP_veto':True,'templates_balanced':True,'masked_background_fill_compared':[127,255],
  'both_fill_scores_used_as_robustness_not_object_pixels':True,'mismatched_object_or_OCR_sensitive':'review/unknown'},
 'binding':{'node_bbox_and_image_hash_required':True,'no_GT_attributes_in_extractor':True,'no_query_ref_in_extractor':True,'no_color_propagation_via_relation_or_fewshot':True,
  'eligible_crop_union_from_predicted_focus_nodes':True,'partial_node_scope_must_be_recorded':True},
 'acceptance':{'output_states':['not_requested','unknown','candidate','accepted_after_version_gate'],
  'unknown_has_reason_code':True,'all_expected_nodes_in_denominator':True,'weak_GT_absent_is_unlabeled':True,
  'minimum_reviewed_accepted_heldout_instances':100,'maximum_human_review_units':200,'required_visual_precision':.90,
  'independent_image_cluster_uncertainty_reported':True,'overall_precision_not_enough_for_unseen_class_certification':True,
  'human_review_covers':['correct_target','mask_contamination','main_color','secondary_color','object_binding','unknown_should_have_been_used'],
  'current_human_reviews_completed':0,'TEST_color_allowed':False},
 'class_rules':perclass,
 'primary_sources':[
  'https://lear.inrialpes.fr/people/vandeweijer/color_names.html','https://lear.inrialpes.fr/people/vandeweijer/papers/NamingTIP09.pdf',
  'https://docs.opencv.org/4.13.0/d8/d83/tutorial_py_grabcut.html','https://github.com/facebookresearch/sam2',
  'https://arxiv.org/abs/2502.04470','https://arxiv.org/abs/1507.00410']}
dest=ROOT/'configs/vg_object_color_rules_v3_20261025.json';assert not dest.exists();dest.write_text(json.dumps(rules,indent=2))
with (ROOT/'reports/VG150_COLOR_CLASS_RULES_V3_20261025.csv').open('w',newline='',encoding='utf-8-sig') as f:
 writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(sorted(csvrows,key=lambda r:r['class_id']))
print(json.dumps({'status':rules['status'],'classes':len(perclass),'not_requested_classes':sum(r['request_priority']=='not_requested' for r in csvrows),'rules_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}))
