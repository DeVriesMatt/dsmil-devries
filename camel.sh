for fold in {1..4}
do
  python train_dsmil.py --fold $fold
done