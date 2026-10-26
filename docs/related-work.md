# Related Work and Existing Solutions

## 1. How the problem is solved in research

**Popularity prediction of online content** is a well-studied task. The usual setup: predict a log-transformed engagement metric (views, likes) for a post from its content, metadata and author features.

| Work | What they did | Relevance to us |
|---|---|---|
| Szabo & Huberman, 2010 [1] | Showed that the long-term popularity of YouTube and Digg content can be predicted from early view counts; popularity is log-normally distributed. | Classic baseline paper. Motivates our log target and our choice of a fixed measurement day (day 14). |
| Pinto, Almeida & Gonçalves, 2013 [2] | Predicted YouTube video popularity from early daily view patterns. | Uses *after-publication* signals. We deliberately use only *pre-publication* features, so a creator can act on the result before posting. |
| Khosla, Das Sarma & Hamid, 2014 [3] | Studied what makes Flickr images popular: content features vs social context (followers, account activity). Social context turned out to be very strong. | Matches our ablation: account context features carry most of the signal. This is why we predict performance *relative to the creator's own level*. |
| Chen et al., 2016 [4] | Popularity prediction for micro-videos (Vine) with visual, audio, text and social features. | Closest content type to TikTok. They focus on multimodal deep learning; we focus on interpretable tabular models. |
| SMP Challenge, 2019–2023 [5, 6] | Annual ACM Multimedia challenge and benchmark (~500K posts, ~70K users) for predicting the popularity of a post *before* it is published. Metrics: Spearman correlation and MAE on log popularity. Top solutions often use gradient boosting (LightGBM, XGBoost) and ensembles. | Same task framing as ours (pre-publication, log target, Spearman). Confirms that tree-based boosting is a strong choice for this task. |
| Grinsztajn et al., 2022 [7] | Benchmark showing tree-based models still outperform deep learning on typical tabular data. | Justifies Random Forest / XGBoost / LightGBM instead of neural networks for our 106 tabular features. |
| Lundberg & Lee, 2017; Lundberg et al., 2020 [8, 9] | SHAP values and TreeSHAP: exact, fast feature attributions for tree ensembles, both per prediction and globally. | Our explainability method. TreeSHAP makes per-creator explanations computationally cheap. |

**What is missing in research:** almost all works train **one global model** for all users and measure accuracy across the whole dataset. The author is treated as a feature (follower count etc.), not as a separate context. Explanations, if present, are global ("what matters on average"), not per creator.

## 2. Industry solutions

| Tool | What it does well | Limitations for our goal |
|---|---|---|
| **vidIQ** | YouTube analytics, keyword research, competitor tracking. Has an LLM-based "AI Coach" connected to the creator's channel data that gives personalized advice. | Focused on YouTube. Advice comes from an LLM reading analytics, not from a model trained and validated on the creator's own videos; there is no measured accuracy or quantified effect of each factor. |
| **TubeBuddy** | YouTube SEO: tag and title suggestions, A/B tests of thumbnails and titles. | YouTube only. A/B tests answer one question at a time (which thumbnail) rather than explaining all factors behind a video's performance. |
| **TikTok Creative Center** | Free hub with trending hashtags, songs, creators and top-performing ads by region and industry. | Built mainly for advertisers. Shows what is popular *in general*, not what works for a specific creator. |
| **TikTok Studio (native analytics)** | Views, watch time, audience demographics per video. | Descriptive only: shows *what* happened, not *why*, and gives no prediction for a new video. |

## 3. Where our project fits

| | Research models | Industry tools | **Our project** |
|---|---|---|---|
| Model per creator | ✗ (one global model) | ✗ / LLM over analytics | ✓ per-creator models (after midterm) + global baseline |
| Target relative to the creator's usual level | rarely | ✗ | ✓ |
| Quantified, validated explanations | global only | ✗ | ✓ SHAP per video and per creator |
| Pre-publication prediction | ✓ | partly | ✓ |
| Open and reproducible | partly | ✗ | ✓ code, data and metrics in this repo |

In short: we combine the **evaluation rigor of research** (held-out test, baselines, significance tests) with the **personal focus of creator tools**, and answer the question these tools don't: *which factors drive views for this particular creator, and how much.*

We also check whether personalization actually helps: the global model is the baseline, and per-creator models must beat it on the same test videos.

## References

1. G. Szabo, B. A. Huberman. *Predicting the popularity of online content.* Communications of the ACM, 53(8), 2010.
2. H. Pinto, J. M. Almeida, M. A. Gonçalves. *Using early view patterns to predict the popularity of YouTube videos.* WSDM, 2013.
3. A. Khosla, A. Das Sarma, R. Hamid. *What makes an image popular?* WWW, 2014.
4. J. Chen, X. Song, L. Nie, X. Wang, H. Zhang, T.-S. Chua. *Micro tells macro: Predicting the popularity of micro-videos via a transductive model.* ACM Multimedia, 2016.
5. B. Wu, W.-H. Cheng, P. Liu, B. Liu, Z. Zeng, J. Luo. *SMP Challenge: An overview of Social Media Prediction Challenge 2019.* ACM Multimedia, 2019. [arXiv:1910.01795](https://arxiv.org/abs/1910.01795)
6. B. Wu et al. *SMP Challenge: An overview and analysis of Social Media Prediction Challenge.* ACM Multimedia, 2023. [arXiv:2405.10497](https://arxiv.org/abs/2405.10497)
7. L. Grinsztajn, E. Oyallon, G. Varoquaux. *Why do tree-based models still outperform deep learning on typical tabular data?* NeurIPS Datasets and Benchmarks, 2022.
8. S. M. Lundberg, S.-I. Lee. *A unified approach to interpreting model predictions.* NeurIPS, 2017.
9. S. M. Lundberg et al. *From local explanations to global understanding with explainable AI for trees.* Nature Machine Intelligence, 2(1), 2020.
10. L. Breiman. *Random forests.* Machine Learning, 45, 2001.
11. T. Chen, C. Guestrin. *XGBoost: A scalable tree boosting system.* KDD, 2016.