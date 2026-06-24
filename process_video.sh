python process_video.py videos/video1.mp4 \
  --output-dir outputs \
  --model dino/checkpoints/tiny_finetune/model_final.pth \
  --valid-model 2022-winners-camera-calibration-challenge/models/model_challenge.pth \
  --valid-diff-threshold 80 \
  --threshold 0.9 \
  --check 10 \
  --frame-step 1 \
  --clahe \
  --verbose
