# train:
python3 cae-mnist.py  --n_epochs=5 --batch_size=128

# test:
python3 cae-mnist.py --cmd test


# model conversion

../tensorflow-pc/bazel-bin/tensorflow/contrib/lite/toco/toco    --input_file=./mnist.pb   --input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE   --output_file=android/app/src/main/assets/mnist.tflite --inference_type=FLOAT   --input_type=FLOAT --input_arrays=x   --output_arrays=output --input_shapes=1,28,28,1


# git command
git commit -a -m "Upload dataset"
git commit README.md -m "Add git command"
git push -u origin master
git pull origin master

# tools 
https://github.com/android-ndk/ndk/wiki
