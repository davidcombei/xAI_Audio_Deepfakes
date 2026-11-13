Audio and explanation masks for three deepfake clips, as .npy files.
Made by export_npys.py (run again with: python3 images_npy/export_npys.py --device cuda:1).

CLIPS
  freq/LJ004-0082_0-1000     0-1 kHz band swapped
  freq/LJ004-0082_3000-4000  3-4 kHz band swapped
  freq/LJ004-0082_6000-7000  6-7 kHz band swapped

FILES IN EACH FOLDER
  audio_original.wav         the wav file, unchanged (22.05 kHz, 9.06 s)
  audio_16k.npy              the waveform the masks describe: (80000,), 16 kHz, first 5 s only
  mask_ground_truth.npy      the known manipulated region, 0 or 1
  mask_saliency.npy          Captum Saliency, raw values (absolute gradients, not scaled to 0-1)
  mask_input_x_gradient.npy  Captum InputXGradient, raw values (can be negative)
  mask_decoder.npy           trained U-Net decoder, 0 to 1

MASK SHAPE
  Masks are (513, 313) = (frequency bins, time frames):
  magnitude of torch.stft(audio_16k, n_fft=1024, hop_length=256, win_length=1024),
  rectangular window, nothing cropped. Row 0 is 0 Hz, row 512 is 8 kHz.

CHECKPOINTS
  freq decoder  ckpts_freqSwapFull_real-fill/UNet_MP_epoch_100_loss_0.2609.pth  (trained with real fill)
  freq detector logReg_ckpts/logReg_vocoded_anyband_controlled.joblib
