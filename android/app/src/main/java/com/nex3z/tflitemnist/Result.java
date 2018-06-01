package com.nex3z.tflitemnist;

import android.graphics.Bitmap;

public class Result {

    private final Bitmap mBitmap;
    private final long mTimeCost;

    public Result(Bitmap bitmap, long timeCost) {
        mBitmap = bitmap;
        mTimeCost = timeCost;
    }

    public Bitmap getBitmap() {
        return mBitmap;
    }

    public long getTimeCost() {
        return mTimeCost;
    }
}
