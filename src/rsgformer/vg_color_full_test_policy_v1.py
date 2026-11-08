"""Frozen predicted-graph-only node selection and unverified color sidecar."""
def select_nodes(graph, eligible, maximum=4):
    endpoints={e[k] for e in graph['edges'][:20] for k in ['subject','object']}
    return sorted((n for n in graph['nodes'] if n['label'] in eligible),
                  key=lambda n:(n['id'] not in endpoints,-n['object_score'],n['id']))[:maximum]

def color_sidecar(graph, selected, records, eligible):
    selected_ids={n['id'] for n in selected};annotations={}
    for node in graph['nodes']:
        identity=node['id']
        if identity not in selected_ids:
            annotations[identity]={'status':'not_requested','reason':'per_image_node_cap' if node['label'] in eligible else 'class_not_in_registered_scope'}
            continue
        row=records[identity];method=row['methods']['SAM2_ColorNames']
        colors=method.get('operating_points',{}).get('conservative',{}).get('colors',[])
        binding=bool(row.get('native_object_binding') and row['native_object_binding']['profile_pass']['strict'])
        proposed=colors if binding else []
        annotations[identity]={'status':'candidate_not_human_validated' if proposed else 'unknown',
            'candidate_colors':proposed,'SAM2_status':method.get('status'),
            'SAM2_reason':method.get('reason'),'native_binding_strict':binding,'human_verified':False}
    return annotations
