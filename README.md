train:
python3 cae-mnist.py --cmd train --n_epochs=1 --batch_size=4096

test:
python3 cae-mnist.py --cmd test

pbToLitr:
../tensorflow-1.8.0/bazel-bin/tensorflow/contrib/lite/toco/toco    --input_file=./mnist.pb   --input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE   --output_file=./mnist.tflite --inference_type=FLOAT   --input_type=FLOAT --input_arrays=x   --output_arrays=output/mul --input_shapes=1,28,28,1
