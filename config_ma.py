L = 5

meta_dir = './0.metadata_model_attribution/'
metadata = [
          "asv19_train_spk_model.txt",
          "asv19_dev_spk_model.txt",
          "asv19_eval_spk_model.txt",
          "asv21_eval_fakes_nocodecs_noasv19_spk_model.txt",
          "asv5_train_spk_model.txt",                                   
          "asv5_dev_spk_model.txt",
          "asv5_eval_nocodecs_spk_model.txt",
          "timit_clean_spk_model.txt",
          "mlaad_v5_spk_model.txt",

]

feats_dir = "./1.features/wav2vec_bert/"

feats = [
        f"wav2vec2-bert_Layer{L}_asv19_train.npy",
        f"wav2vec2-bert_Layer{L}_asv19_dev.npy",
        f"wav2vec2-bert_Layer{L}_asv19_eval.npy",
        f"wav2vec2-bert_Layer{L}_asv21.npy",
        f"wav2vec2-bert_Layer{L}_asv5_train.npy",
        f"wav2vec2-bert_Layer{L}_asv5_dev.npy",
        f"wav2vec2-bert_Layer{L}_asv5_eval_nocodec.npy",
        f"wav2vec2-bert_Layer{L}_timit_tts_clean_new.npy",
        f"wav2vec2-bert_Layer{L}_mlaad_v5.npy",
    ]





