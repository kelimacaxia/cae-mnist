package com.nex3z.tflitemnist;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.ref.SoftReference;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class Classifier {
    private static final String LOG_TAG = Classifier.class.getSimpleName();

    private static final String MODEL_PATH = "mnist.tflite";

    private static final int DIM_BATCH_SIZE = 1;
    public static final int DIM_IMG_SIZE_HEIGHT = 28;
    public static final int DIM_IMG_SIZE_WIDTH = 28;
    private static final int DIM_PIXEL_SIZE = 1;
    private static final int CATEGORY_COUNT = 10;

    private final Interpreter mTfLite;
    private final ByteBuffer mImgData;
    private final int[] mImagePixels = new int[DIM_IMG_SIZE_HEIGHT * DIM_IMG_SIZE_WIDTH];
    private final float[][] mResult = new float[1][28*28];

    Classifier(Activity activity) throws IOException {
        MappedByteBuffer buffer = loadModelFile(activity);
        mTfLite = new Interpreter(buffer);

        mImgData = ByteBuffer.allocateDirect(
                4 * DIM_BATCH_SIZE * DIM_IMG_SIZE_HEIGHT * DIM_IMG_SIZE_WIDTH * DIM_PIXEL_SIZE);
        mImgData.order(ByteOrder.nativeOrder());
    }

    public Result classify(Bitmap bitmap) {
        mResult[0][0] = 0.3f;
        convertBitmapToByteBuffer(bitmap);
        long startTime = SystemClock.uptimeMillis();
        mTfLite.run(mImgData, mResult);
        long endTime = SystemClock.uptimeMillis();
        long timeCost = endTime - startTime;
        Log.i(LOG_TAG, "run(): result = " + Arrays.toString(mResult[0])
                + ", timeCost = " + timeCost);
        return new Result(convertFloatBufferToBitmap(mResult[0]), timeCost);
    }

    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (mImgData == null) {
            return;
        }
        mImgData.rewind();

        bitmap.getPixels(mImagePixels, 0, bitmap.getWidth(), 0, 0,
                bitmap.getWidth(), bitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_WIDTH; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_HEIGHT; ++j) {
                final int val = mImagePixels[pixel++];
                mImgData.putFloat(convertToGreyScale(val));
            }
        }
    }

    private float convertToGreyScale(int color) {
        return (((color >> 16) & 0xFF) + ((color >> 8) & 0xFF) + (color & 0xFF)) / 3.0f / 255.0f;
    }

    private Bitmap convertFloatBufferToBitmap(float[] data) {
        int[] imgInt = new int[28 * 28];

        for (byte y = 0; y < 28; y++)
            for (byte x = 0; x < 28 ; x++) {
                byte pixel = (byte)(data[y * 28 + x]*255);
                imgInt[y * 28 + x] = 0xFF000000|pixel<<16| pixel<<8|pixel;
            }

        Bitmap bitmap= Bitmap.createBitmap(28, 28, Bitmap.Config.ARGB_8888);
        bitmap.setPixels(imgInt,0, 28, 0, 0, 28, 28);

        return bitmap;
    }

}
