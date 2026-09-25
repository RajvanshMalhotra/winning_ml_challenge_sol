# Blocking channels: recall alone, combined, and marginal gain (train)

Marginal gain = recall lost if that one channel is removed while all others stay.

| country | channel | candidates per S1 | recall alone | marginal gain |
|---|---|---|---|---|
| ALL | tfidf top-50 | 50.0 | 95.82% | +0.97% |
| ALL | key blocks | 30.1 | 87.08% | +0.37% |
| ALL | name-only | 19.6 | 3.92% | +0.24% |
| ALL | qwen hard-names | 5.0 | 4.22% | +0.24% |
| ALL | graph expansion | 32.2 | 93.89% | +0.56% |
| ALL | **ALL CHANNELS** | 107.6 | **98.28%** | |
| India | tfidf top-50 | 50.0 | 92.27% | +2.17% |
| India | key blocks | 22.0 | 77.70% | +0.53% |
| India | name-only | 19.4 | 3.49% | +0.38% |
| India | qwen hard-names | 9.9 | 5.24% | +0.37% |
| India | graph expansion | 37.6 | 90.19% | +1.22% |
| India | **ALL CHANNELS** | 114.5 | **96.46%** | |
| US | tfidf top-50 | 50.0 | 98.19% | +0.18% |
| US | key blocks | 35.4 | 93.35% | +0.26% |
| US | name-only | 19.7 | 4.20% | +0.14% |
| US | qwen hard-names | 1.7 | 3.53% | +0.15% |
| US | graph expansion | 28.6 | 96.37% | +0.12% |
| US | **ALL CHANNELS** | 103.0 | **99.51%** | |
