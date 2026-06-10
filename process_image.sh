python process_image.py images/image1.png \
  --output-dir results \
  --model dino/checkpoints/tiny_finetune/model_final.pth \
  --valid-model 2022-winners-camera-calibration-challenge/models/model_challenge.pth \
  --valid-diff-threshold 80 \
  --threshold 0.9 \
  --check 10 \
  --clahe --draw-keypoints --print-coords \
  --verbose