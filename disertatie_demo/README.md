# disertatie_demo

Self-contained local demo of the full pipeline: an uploaded speech clip is
scored by the detector; if it is real the pipeline stops; if it is fake, a k-NN
attributes it to its most likely generator and a U-Net highlights the
time-frequency regions the detector relies on.

Everything needed to run lives in this directory. The one runtime download is
facebook/w2v-bert-2.0 (fetched from the Hugging Face Hub on the first fake clip
and cached locally).

## Pipeline

1. Detection: wav2vec2-xls-r-2b (layer 9, mean-pooled) with logistic regression.
   Real vs. fake. Real clips stop here.
2. Attribution: k-NN (k=11) over facebook/w2v-bert-2.0 layer-5 features across
   243 generator systems (ASVspoof 2019/2021/5, TIMIT-TTS, MLAAD v5).
3. Explainability: U-Net relevance mask over the log-magnitude spectrogram,
   with relevant / irrelevant reconstructions re-scored as a faithfulness check.

## Layout

    streamlit_demo.py                      Streamlit app with an upload button
    pipeline.py                            detector, k-NN attribution, U-Net
    config.py audio.py models.py viz.py    supporting code
    requirements.txt
    checkpoints/knn_model_attribution.joblib          k-NN, scaler, label map
    ckpts/logReg_ckpts/logReg_MLAAD_M-AILabs.joblib   detector head
    ckpts/ckpts_realisticDeepfakes/UNet_MP_epoch_41_loss_0.3316.pth   U-Net mask
    models/wav2vec2-xls-r-2b_truncated/    detector backbone (layers 0-8)
    examples/fake examples/real            sample clips

## Run

    python -m streamlit run streamlit_demo.py

Open the printed URL, upload an audio file (or pick a bundled example from the
sidebar), and click Analyze.

The sidebar exposes a compute-device selector (cpu, cuda and mps are offered
when available) and the detection threshold. On Apple Silicon select mps;
PYTORCH_ENABLE_MPS_FALLBACK is set so any op not yet implemented on MPS runs on
CPU. On CPU a single clip takes about 30-60 s.

## Environment overrides

- KNN_CKPT - path to the k-NN checkpoint (default: checkpoints/knn_model_attribution.joblib).
- DETECT_THRESHOLD - P(fake) above which a clip is called fake (default 0.5).
- ADDVISOR_REPO_ROOT - where config/audio/models/viz live (default: this directory).

## Notes

The detector backbone here is the truncated xls-r-2b (only the layers up to the
one the detector reads), with num_hidden_layers set to 9 so it loads light and
fast while producing the same layer-9 features as the full model.

The k-NN checkpoint is prebuilt. Rebuilding it requires the full research repo
(the pre-extracted 1.features/wav2vec_bert/ layer-5 features and config_ma.py)
and is done there with run_model_attribution_new.py.
