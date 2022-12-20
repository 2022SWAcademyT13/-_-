// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.pytorch.demo.objectdetection;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Rect;
import android.util.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Objects;

class Result {
    int classIndex;
    Float score;
    Rect rect;

    public Result(int cls, Float output, Rect rect) {
        this.classIndex = cls;
        this.score = output;
        this.rect = rect;
    }
};

public class PrePostProcessor {
    // for yolov5 model, no need to apply MEAN and STD
    public final static float[] NO_MEAN_RGB = new float[] {0.0f, 0.0f, 0.0f};
    public final static float[] NO_STD_RGB = new float[] {1.0f, 1.0f, 1.0f};
    // model input image size
    public final static int INPUT_WIDTH = 550;
    public final static int INPUT_HEIGHT = 550;
    public final static int OUTPUT_COLUMN = 7; // left, top, right, bottom, score, label, index

    static String[] mClasses; // class labels ; person, dog, ...


    static ArrayList<Result> outputsToPredictions(int countResult, float[] outputs, float imgScaleX, float imgScaleY, float ivScaleX, float ivScaleY, float startX, float startY) {
        ArrayList<Result> results = new ArrayList<>();
        for (int i = 0; i< countResult; i++) {
            float left = outputs[i* OUTPUT_COLUMN];
            float top = outputs[i* OUTPUT_COLUMN +1];
            float right = outputs[i* OUTPUT_COLUMN +2];
            float bottom = outputs[i* OUTPUT_COLUMN +3];

            left = imgScaleX * left;
            top = imgScaleY * top;
            right = imgScaleX * right;
            bottom = imgScaleY * bottom;

            Rect rect = new Rect((int)(startX+ivScaleX*left), (int)(startY+top*ivScaleY), (int)(startX+ivScaleX*right), (int)(startY+ivScaleY*bottom));
            Result result = new Result((int)outputs[i* OUTPUT_COLUMN +5], outputs[i* OUTPUT_COLUMN +4], rect);
            results.add(result);
        }
        return results;
    }

    static ArrayList<Float> nonMaxSuppresion(int countResult, ArrayList<Float> outputs, double iouthreshold) {
//        Log.v("YolactPrePostNMS", "NMS start");
//        Log.v("YolactPrePostNMS", "countResult: " + countResult);
//        Log.v("YolactPrePostNMS", "outputs: " + outputs);
//        Log.v("YolactPrePostNMS", "iouthreshold: " + iouthreshold);
        ArrayList<Float> results = new ArrayList<>();
        ArrayList<Float> result;
        ArrayList<Float> lastresult = null;
        float maxconf;
        int maxi;
        while (countResult > 0) {
            if (lastresult != null) {
//                Log.v("YolactPrePostNMS", "lastresult: " + lastresult);
                int count= countResult;
                for (int i=0; i < countResult; i++) {
//                    Log.v("YolactPrePostNMS", "i: " + i);
                    // 한번 돌면서 저번거와 클래스 같고 iou 큰거 없애기
                    if (Objects.equals(outputs.get(i * 7 + 5),lastresult.get(5)))  {
                        float x11 = outputs.get(i * 7 + 0);// 서브리스트 또는 컷 메소드
                        float y11 = outputs.get(i * 7 + 1);// 순수 자바 NMS 찾아보기
                        float x12 = outputs.get(i * 7 + 2);
                        float y12 = outputs.get(i * 7 + 3);

                        float x21 = lastresult.get(0);
                        float y21 = lastresult.get(1);
                        float x22 = lastresult.get(2);
                        float y22 = lastresult.get(3);

                        double a1 = (x11 - x12) * (y11 - y12);
                        double a2 = (x21 - x22) * (y21 - y22);

                        float x31 = Math.max(x11, x21);
                        float y31 = Math.max(y11, y21);
                        float x32 = Math.min(x12, x22);
                        float y32 = Math.min(y12, y22);

                        float w = Math.max(x32 - x31, 0);
                        float h = Math.max(y32 - y31, 0);

                        double a3 = w * h;
                        double Iou = a3 / (a1 + a2 - a3);
//                        Log.v("YolactPrePostNMS", "IOU: " + Iou);
                        if (Iou > iouthreshold) {
                            for (int j=0; j < 7; j++) {
                                outputs.remove(i * 7);
                            }
                            countResult -= 1;
                            i -= 1;
//                            Log.v("YolactPrePostNMS", "remove one, IOU: " + Iou);
//                            Log.v("YolactPrePostNMS", "output len: " + outputs.size());
                        }
                    }
                }
            }
            // 최대확률 하나 뽑아서 넣기
            result = new ArrayList<Float>();
            if (countResult > 0) {
//                Log.v("YolactPrePostNMS", "keep one");
                maxconf = -Float.MAX_VALUE;
                maxi = -1;
                for (int i = 0; i < countResult; i++) {
                    if (outputs.get(i * 7 + 4) > maxconf) {
                        maxconf = outputs.get(i * 7 + 4);
                        maxi = i;
                    }
                }
                for (int i = 0; i < 7; i++) {
                    float temp = outputs.remove(maxi * 7);
                    result.add(temp);
                }
//                Log.v("YolactPrePostNMS", "result: " + result);
                countResult -= 1;
                lastresult = result;
                results.addAll(result);
//                Log.v("YolactPrePostNMS", "outputs len after a loop: " + outputs.size());
//                Log.v("YolactPrePostNMS", "countResult after a loop: " + countResult);
            }
        }
//        Log.v("YolactPrePostNMS", "results len : "+ results.size());
        return results;
    }

    static ArrayList<Integer> closerange(int count, ArrayList<Float> outputs, float maxY) {
        if (count > 0) {
            ArrayList<Integer> results = new ArrayList<Integer>();
            for (int i = 0; i < count; i++) {
                float Y = outputs.get(i * 7 + 3);
                if (Y > maxY) {
                    results.add(outputs.get(i * 7 + 5).intValue());
                }
            }
            return results;
        }
        ArrayList<Integer> results = null;
        return results;
    }

    static String redgreen(Result result, Bitmap bitmap, int ivScaleX, int ivScaleY) {
        if (bitmap == null) return "null";
        String color = "?";
        Bitmap cropped = Bitmap.createBitmap(bitmap, (int) (result.rect.left*bitmap.getWidth()/ivScaleX), (int) (result.rect.top*bitmap.getHeight()/ivScaleY), (int) (result.rect.width()*bitmap.getWidth()/ivScaleX), (int) (result.rect.height()*bitmap.getHeight()/ivScaleY));
        for (int i=0; i < cropped.getWidth(); i++) {
            for (int j=0; j < cropped.getHeight(); j++) {
                int pixel = cropped.getPixel(i,j);
                int r = Color.red(pixel);
                int g = Color.green(pixel);
                int b = Color.blue(pixel);
                if (r>128 & g<128 & b<128) {
                    color = "red";
                    break;
                } else if (r<100 & g>128) {
                    color = "green";
                    break;
                }
            }
        }
        return color;
    }
}
