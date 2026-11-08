import json, os
from collections import Counter
import re

p = os.path.join(os.path.expanduser("~"), "rsg-former/data/ucm_captions/dataset.json")
data = json.load(open(p))
train_imgs = [img for img in data['images'] if img.get('split') == 'train']
print(f"Total train images: {len(train_imgs)}")

all_sentences = []
for img in train_imgs:
    for s in img.get('sentences', []):
        all_sentences.append(s.get('raw', '').strip())

print(f"Total train sentences: {len(all_sentences)}")
print("\nTop 20 most frequent exact sentences in UCM train split:")
sentence_counts = Counter(all_sentences)
for s, c in sentence_counts.most_common(20):
    print(f"  [{c} times] {s}")

# Check unigram frequencies
words = []
for s in all_sentences:
    toks = re.findall(r'\w+', s.lower())
    words.extend(toks)

print("\nTop 20 most frequent words in UCM train:")
word_counts = Counter(words)
for w, c in word_counts.most_common(20):
    print(f"  {w}: {c}")

# Check length distribution
lengths = [len(s.split()) for s in all_sentences]
print(f"\nAverage sentence length: {sum(lengths)/len(lengths):.1f} words (min={min(lengths)}, max={max(lengths)})")

# Check class distribution / categories
# UC Merced has 21 categories: agricultural, airplane, baseballdiamond, beach, buildings, chaparral, denseresidential, forest, freeway, golfcourse, harbor, intersection, mediumresidential, mobilehomepark, overpass, parkinglot, river, runway, sparseresidential, storagetanks, tenniscourt
print("\nCheck some sample filenames:")
for img in train_imgs[:10]:
    print(f"  {img.get('filename')}: {img.get('sentences')[0]['raw']}")

