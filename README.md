# train:
python3 cae-mnist.py  --n_epochs=5 --batch_size=128

# test:
python3 cae-mnist.py --cmd test


# model conversion

../tensorflow/bazel-bin/tensorflow/contrib/lite/toco/toco   --input_file=./cae-mnist.pb   --input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE   --output_file=android/app/src/main/assets/mnist.tflite --inference_type=FLOAT   --input_type=FLOAT --input_arrays=x   --output_arrays=output --input_shapes=1,28,28,1




python3 rnn-mnist.py --cmd train

python3 rnn-mnist.py --cmd test

../tensorflow/bazel-bin/tensorflow/contrib/lite/toco/toco   --input_file=./rnn-mnist.pb   --input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE   --output_file=android/app/src/main/assets/mnist.tflite --inference_type=FLOAT   --input_type=FLOAT --input_arrays=x   --output_arrays=output --input_shapes=1,28,28

# git command
git commit -a -m "Upload dataset"
git commit *.py -m "Use conv2d  to repalce conv2d_transpos , because tflite doesn't support conv2d_transpos"
git push -u origin master
git pull origin master

# tools 
https://github.com/android-ndk/ndk/wiki
