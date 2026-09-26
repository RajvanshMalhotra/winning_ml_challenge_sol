# Bi-encoder bake-off

| model | tag | India R@10 | India R@50 | India R@100 | India-native R@10 | India-native R@50 | India-native R@100 | US R@10 | US R@50 | US R@100 | rec/s | test vec GB | train peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bgem3 | finetune | 0.9969 | 0.9995 | 0.9998 | 0.9999 | 1.0000 | 1.0000 | 0.9981 | 0.9997 | 0.9999 | 2352 | 20.4 | 13.6 |
| bgem3 | finetune_US | 0.9797 | 0.9921 | 0.9945 | 0.9566 | 0.9857 | 0.9911 | 0.9982 | 0.9997 | 0.9999 | 3500 | 20.4 | 12.1 |
| bgem3 | zeroshot | 0.9076 | 0.9539 | 0.9654 | 0.7933 | 0.9157 | 0.9478 | 0.9689 | 0.9829 | 0.9864 | 1809 | 20.4 | 0.0 |

Winner: **bgem3** (mean fine-tuned R@50; ties within 0.005 broken by leave-one-country-out India R@50, then throughput)
