// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.pytorch.demo.objectdetection;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;

import java.util.ArrayList;


public class ResultView extends View {

    private final static int TEXT_X = 40;
    private final static int TEXT_Y = 35;
    private final static int TEXT_WIDTH = 260;
    private final static int TEXT_HEIGHT = 50;

    private Paint mPaintRectangle;
    private Paint mPaintText;
    private ArrayList<Result> mResults;
    private Bitmap mmBitmap;
    private Bitmap moverlay;

    public ResultView(Context context) {
        super(context);
    }

    public ResultView(Context context, AttributeSet attrs){
        super(context, attrs);
        mPaintRectangle = new Paint();
        mPaintRectangle.setColor(Color.YELLOW);
        mPaintText = new Paint();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if (mResults == null) return;
        for (Result result : mResults) {
            mPaintRectangle.setStrokeWidth(5);
            mPaintRectangle.setStyle(Paint.Style.STROKE);
            canvas.drawRect(result.rect, mPaintRectangle);
            // 텍스트 둘러싼 상자
            Path mPath = new Path();
            RectF mRectF = new RectF(result.rect.left, result.rect.top, result.rect.left + TEXT_WIDTH,  result.rect.top + TEXT_HEIGHT);
            mPath.addRect(mRectF, Path.Direction.CW);
            mPaintText.setColor(Color.MAGENTA);
            canvas.drawPath(mPath, mPaintText);
            // 텍스트 설명
            mPaintText.setColor(Color.WHITE);
            mPaintText.setStrokeWidth(0);
            mPaintText.setStyle(Paint.Style.FILL);
            mPaintText.setTextSize(32);
            if (result.classIndex == 9) {
                canvas.drawText(String.format("%s %s %.2f", PrePostProcessor.mClasses[result.classIndex], PrePostProcessor.redgreen(result, mmBitmap, moverlay.getWidth(), moverlay.getHeight()), result.score), result.rect.left + TEXT_X, result.rect.top + TEXT_Y, mPaintText);
            } else {
                canvas.drawText(String.format("%s %.2f", PrePostProcessor.mClasses[result.classIndex], result.score), result.rect.left + TEXT_X, result.rect.top + TEXT_Y, mPaintText);
            }
        }
        float startx = (float) (canvas.getWidth() - moverlay.getWidth())/2;
        float starty = (float) (canvas.getHeight() - moverlay.getHeight())/2;
        canvas.drawBitmap(moverlay, startx, starty,new Paint(Paint.FILTER_BITMAP_FLAG));
//        Log.v("YolactResultView", "canvas height: "+ canvas.getHeight());
//        Log.v("YolactResultView", "starty: "+ starty);
//        Log.v("YolactResultView", "moverlay width: "+ moverlay.getWidth());
//        Log.v("YolactResultView", "moverlay height: "+ moverlay.getHeight());
    }

    public void setResults(ArrayList<Result> results) {
        mResults = results;
    }
    public void setOverlay(Bitmap bitmap) {
        moverlay = bitmap;
    }

    public void setmBitmap(Bitmap bitmap) {
        mmBitmap = bitmap;
    }
}
