package com.nex3z.tflitemnist;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.util.Random;

import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnClick;

public class MainActivity extends AppCompatActivity {
    private static final String LOG_TAG = MainActivity.class.getSimpleName();

    @BindView(R.id.fpv_paint) FingerPaintView mFpvPaint;
    @BindView(R.id.tv_timecost) TextView mTvTimeCost;
    @BindView(R.id.noise_image_view) ImageView mNosiseImageView;
    @BindView(R.id.processed_image_view) ImageView mProcessedImageView;
    private Classifier mClassifier;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ButterKnife.bind(this);
        init();
    }

    @OnClick(R.id.btn_process)
    void onDetectClick() {
        if (mClassifier == null) {
            Log.e(LOG_TAG, "onDetectClick(): Classifier is not initialized");
            return;
        } else if (mFpvPaint.isEmpty()) {
            Toast.makeText(this, R.string.please_write_a_digit, Toast.LENGTH_SHORT).show();
            return;
        }

        Bitmap image = mFpvPaint.exportToBitmap(
                Classifier.DIM_IMG_SIZE_WIDTH, Classifier.DIM_IMG_SIZE_HEIGHT);
        Bitmap inverted = ImageUtil.invert(image);

        Random random = new Random();
        for (int y=0; y<inverted.getHeight(); y++)
            for (int x=0; x<inverted.getWidth(); x++) {
                int noise = random.nextInt(64);
                int color = 0xFF000000| noise<<16|noise<<8| noise;
                if ((inverted.getPixel(x,y) & 0xFF) < 30)
                    inverted.setPixel(x, y, color);
            }

        mNosiseImageView.setImageBitmap(inverted);
        Result result = mClassifier.classify(inverted);
        Bitmap processed = result.getBitmap();
        mProcessedImageView.setImageBitmap(processed);
        mTvTimeCost.setText(String.format(getString(R.string.timecost_value),
                result.getTimeCost()));
    }

    @OnClick(R.id.btn_clear)
    void onClearClick() {
        mFpvPaint.clear();
        mTvTimeCost.setText(R.string.empty);
    }

    private void init() {
        try {
            mClassifier = new Classifier(this);
        } catch (IOException e) {
            Toast.makeText(this, R.string.failed_to_create_classifier, Toast.LENGTH_LONG).show();
            Log.e(LOG_TAG, "init(): Failed to create tflite model", e);
        }
    }
}
