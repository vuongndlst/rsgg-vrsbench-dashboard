"""Centralized Dataset Registry for Remote Sensing Captioning & Grounding.
Specifies prompt templates, few-shot exemplars, style profiles, and decoding parameters
for each dataset without modifying model weights (Zero-Shot / In-Domain Transfer).
"""

DATASET_REGISTRY = {
    "vrsbench": {
        "dataset_name": "VRSBench",
        "task_type": "detailed_captioning_and_grounding",
        "domain_description": "High-resolution remote sensing image captioning with spatial grounding",
        "target_word_range": (18, 24),
        "target_word_count": 21,
        "max_new_tokens": 48,
        "min_new_tokens": 16,
        "beam_size": 3,
        "length_penalty": 0.90,
        "no_repeat_ngram_size": 3,
        "box_confidence_threshold": 0.25,
        "max_prompt_boxes": 12,
        "few_shot_exemplars": [
            {
                "detected_objects": "planes (center, left), runway (background)",
                "caption": "There are several planes parked on the apron next to the runway."
            },
            {
                "detected_objects": "houses (dense), trees (surrounding), roads (crossing)",
                "caption": "The image shows a dense residential area with many houses surrounded by green trees and paved roads."
            },
            {
                "detected_objects": "ships (docked), port facilities (waterfront)",
                "caption": "Several large cargo ships are docked along the piers at a busy seaport."
            }
        ],
        "system_instruction": (
            "You are an expert remote sensing image analyst. Based strictly on the visual scene graph "
            "and detected entities with spatial locations, generate a natural, concise, factual caption. "
            "Describe the overall scene, prominent objects, and their spatial arrangements in exactly 18 to 24 words. "
            "Do not hallucinate objects not present in the scene graph."
        )
    },
    "sydney": {
        "dataset_name": "Sydney-Captions",
        "task_type": "concise_captioning",
        "domain_description": "Coastal and urban remote sensing imagery in Sydney",
        "target_word_range": (10, 16),
        "target_word_count": 13,
        "max_new_tokens": 30,
        "min_new_tokens": 8,
        "beam_size": 3,
        "length_penalty": 0.90,
        "no_repeat_ngram_size": 3,
        "box_confidence_threshold": 0.20,
        "max_prompt_boxes": 8,
        "few_shot_exemplars": [
            {
                "detected_objects": "4 swimming-pools around (top-left, center, bottom-right); 3 vehicles around (center, left)",
                "caption": "A residential area with houses arranged neatly and some roads go through this area."
            },
            {
                "detected_objects": "2 swimming-pools around (center); tennis-court at (top-right); vehicles",
                "caption": "Lots of houses with red and orange roofs arranged neatly in lines."
            },
            {
                "detected_objects": "one airplane at (center); one vehicle at (bottom)",
                "caption": "A white airplane parked on the airport with some runways beside."
            },
            {
                "detected_objects": "No prominent foreground objects detected (open landscape or general terrain).",
                "caption": "This is a part of deep green sparkling sea."
            },
            {
                "detected_objects": "2 ships around (center, left); water surface",
                "caption": "A wide river with deep green waters and some boats on it."
            },
            {
                "detected_objects": "2 storage-tanks around (center); vehicles around (bottom)",
                "caption": "An industrial area with many white buildings and some roads go through this area."
            }
        ],
        "system_instruction": (
            "You are an expert remote sensing aerial interpreter specializing in Sydney aerial imagery. "
            "Based on the detected entities and scene layout, generate exactly one factual sentence in the "
            "canonical Sydney-Captions style (10 to 15 words) and end with a period. "
            "Domain guidelines: "
            "1. Clusters of swimming pools, tennis courts, vehicles, or residential lots represent: "
            "'A residential area with houses arranged neatly and some roads go through this area.' or "
            "'Lots of houses with red and orange roofs arranged neatly.' "
            "2. Airplanes or runways represent: 'A white airplane parked on the airport with some runways beside.' "
            "3. Ships, harbors, or open water represent: 'A wide river with deep green waters and some boats on it.' "
            "4. Empty terrain with no prominent objects represents: 'This is a part of deep green sparkling sea.' "
            "5. Storage tanks or industrial structures represent: 'An industrial area with many white buildings and some roads go through this area.' "
            "Never output introductory commentary; generate only the single descriptive sentence."
        )
    },
    "ucm": {
        "dataset_name": "UCM-Captions",
        "task_type": "land_use_captioning",
        "domain_description": "UC Merced 21-class land-use aerial scenes",
        "target_word_range": (9, 14),
        "target_word_count": 11,
        "max_new_tokens": 28,
        "min_new_tokens": 7,
        "beam_size": 3,
        "length_penalty": 0.85,
        "no_repeat_ngram_size": 3,
        "box_confidence_threshold": 0.20,
        "max_prompt_boxes": 8,
        "few_shot_exemplars": [
            {
                "detected_objects": "many vehicles in parking lot, regular grid",
                "caption": "Lots of cars parked in lines in the parking lot."
            },
            {
                "detected_objects": "baseball_field (center), manicured grass outfield",
                "caption": "A regular baseball diamond compose of manicured lawns and sand."
            },
            {
                "detected_objects": "airplane (center), airport runway, tarmac",
                "caption": "A straight runway with some white mark lines and airplanes parked beside."
            },
            {
                "detected_objects": "ships around (harbor), docks, waterfront",
                "caption": "Lots of boats docked in lines at the harbor."
            },
            {
                "detected_objects": "2 tennis_courts arranged neatly on lawn",
                "caption": "Two tennis courts arranged neatly with some plants surrounded."
            },
            {
                "detected_objects": "circular storage_tanks in facility",
                "caption": "Three different storage tanks are on the ground with some buildings beside."
            },
            {
                "detected_objects": "overpass crossing freeway with roads",
                "caption": "An overpass with a road go across another roads vertically."
            },
            {
                "detected_objects": "No prominent foreground objects detected (open landscape or general terrain).",
                "caption": "Lots of plants scattered on the ground."
            },
            {
                "detected_objects": "swimming_pools, houses, vehicles in residential grid",
                "caption": "A dense residential area with many houses arranged neatly and some roads go through."
            }
        ],
        "system_instruction": (
            "You are an expert remote sensing aerial analyst specializing in UC Merced Land-Use captioning. "
            "Based strictly on detected objects and visual terrain, output exactly one factual sentence in the "
            "canonical UCM-Captions style (9 to 13 words) ending with a period. "
            "Domain guidelines: "
            "1. Vehicles in rows -> 'Lots of cars parked in lines in the parking lot.' "
            "2. Baseball field -> 'A regular baseball diamond compose of manicured lawns and sand.' "
            "3. Airplanes / runways -> 'A straight runway with some white mark lines on it.' "
            "4. Harbor / ships -> 'Lots of boats docked in lines at the harbor.' "
            "5. Tennis courts -> 'Two tennis courts arranged neatly with some plants surrounded.' "
            "6. Storage tanks -> 'Three different storage tanks are on the ground with some buildings beside.' "
            "7. Overpass / freeway -> 'An overpass with a road go across another roads vertically.' "
            "8. Open vegetation / forest / chaparral / farmland -> 'Lots of plants scattered on the ground.' or 'There is a piece of farmland.' "
            "9. Suburban / residential lots -> 'A dense residential area with many houses arranged neatly and some roads go through.' "
            "Keep the phrasing strictly within 9 to 13 words without extra commentary."
        )
    },
    "rsicd": {
        "dataset_name": "RSICD",
        "task_type": "remote_sensing_captioning",
        "domain_description": "Remote Sensing Image Captioning Dataset with 30 landscape categories",
        "target_word_range": (8, 14),
        "target_word_count": 11,
        "max_new_tokens": 28,
        "min_new_tokens": 6,
        "beam_size": 3,
        "length_penalty": 0.85,
        "no_repeat_ngram_size": 3,
        "box_confidence_threshold": 0.20,
        "max_prompt_boxes": 8,
        "few_shot_exemplars": [
            {
                "detected_objects": "airplanes parked near terminal and runways",
                "caption": "Many planes are parked in an airport near many runways."
            },
            {
                "detected_objects": "stadium with green field surrounded by urban area",
                "caption": "A stadium with a green sports field in an urban area."
            },
            {
                "detected_objects": "baseball diamond with sand and green grass",
                "caption": "A baseball field surrounded by trees and residential houses."
            },
            {
                "detected_objects": "ships docked at harbor near water",
                "caption": "Many boats are docked in a busy port."
            },
            {
                "detected_objects": "bridge crossing wide river with vegetation",
                "caption": "A long bridge goes across a river with green trees on both sides."
            },
            {
                "detected_objects": "storage tanks in industrial area",
                "caption": "Several circular storage tanks are in an industrial zone."
            }
        ],
        "system_instruction": (
            "You are an expert remote sensing aerial interpreter. Based on detected visual objects "
            "and land features, output exactly one concise factual sentence in the canonical RSICD style "
            "(8 to 13 words) ending with a period. "
            "Never output introductory commentary; generate only the single descriptive sentence."
        )
    },
    "dior_rsvg": {
        "dataset_name": "DIOR-RSVG",
        "task_type": "referring_expression_grounding",
        "domain_description": "Visual Grounding with text query on DIOR remote sensing images",
        "box_confidence_threshold": 0.15,
        "iou_thresholds": [0.5, 0.7, 0.75],
        "system_instruction": (
            "Given a remote sensing image and a natural language referring expression query, "
            "locate the exact bounding box [ymin, xmin, ymax, xmax] of the referred target object."
        )
    }
}

def get_dataset_config(dataset_name: str) -> dict:
    key = dataset_name.lower().replace("-", "_")
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[key]

