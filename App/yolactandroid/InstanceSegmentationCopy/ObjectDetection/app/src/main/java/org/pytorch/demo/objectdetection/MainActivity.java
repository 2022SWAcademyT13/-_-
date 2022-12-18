// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.pytorch.demo.objectdetection;

import android.Manifest;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.StringReader;
import java.lang.reflect.Array;
import java.nio.FloatBuffer;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;


public class MainActivity extends AppCompatActivity implements Runnable {

static {
    if (!NativeLoader.isInitialized()) {
        NativeLoader.init(new SystemDelegate());
    }
//    NativeLoader.loadLibrary("pytorch_jni");
    NativeLoader.loadLibrary("torchvision_ops");
}


    private int mImageIndex = 0;
    private final String[] mTestImages = {"test1.png", "test2.jpg", "test3.png", "dog550.png"};

    private ImageView mImageView;
    private ResultView mResultView;
    private Button mButtonDetect;
    private ProgressBar mProgressBar;
    private Bitmap mBitmap = null;
    private Module mModule = null;
    private float mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY;

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.v("Yolact", "onCreate");
        super.onCreate(savedInstanceState);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 1);
        }

        setContentView(R.layout.activity_main);

        try {
            mBitmap = BitmapFactory.decodeStream(getAssets().open(mTestImages[mImageIndex]));
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }

        mImageView = findViewById(R.id.imageView);
        mImageView.setImageBitmap(mBitmap);
        mResultView = findViewById(R.id.resultView);
        mResultView.setVisibility(View.INVISIBLE);

        final Button buttonTest = findViewById(R.id.testButton);
        buttonTest.setText(("Test Image 1/4"));
        buttonTest.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mResultView.setVisibility(View.INVISIBLE);
                mImageIndex = (mImageIndex + 1) % mTestImages.length;
                buttonTest.setText(String.format("Text Image %d/%d", mImageIndex + 1, mTestImages.length));

                try {
                    mBitmap = BitmapFactory.decodeStream(getAssets().open(mTestImages[mImageIndex]));
                    mImageView.setImageBitmap(mBitmap);
                } catch (IOException e) {
                    Log.e("Object Detection", "Error reading assets", e);
                    finish();
                }
            }
        });


        final Button buttonSelect = findViewById(R.id.selectButton);
        buttonSelect.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mResultView.setVisibility(View.INVISIBLE);

                final CharSequence[] options = { "Choose from Photos", "Take Picture", "Cancel" };
                AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
                builder.setTitle("New Test Image");

                builder.setItems(options, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int item) {
                        if (options[item].equals("Take Picture")) {
                            Intent takePicture = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                            startActivityForResult(takePicture, 0);
                        }
                        else if (options[item].equals("Choose from Photos")) {
                            Intent pickPhoto = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.INTERNAL_CONTENT_URI);
                            startActivityForResult(pickPhoto , 1);
                        }
                        else if (options[item].equals("Cancel")) {
                            dialog.dismiss();
                        }
                    }
                });
                builder.show();
            }
        });

        final Button buttonLive = findViewById(R.id.liveButton);
        buttonLive.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
              final Intent intent = new Intent(MainActivity.this, ObjectDetectionActivity.class);
              startActivity(intent);
            }
        });

        mButtonDetect = findViewById(R.id.detectButton);
        mProgressBar = (ProgressBar) findViewById(R.id.progressBar);
        mButtonDetect.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mButtonDetect.setEnabled(false);
                mProgressBar.setVisibility(ProgressBar.VISIBLE);
                mButtonDetect.setText(getString(R.string.run_model));

//                mImgScaleX = (float)mBitmap.getWidth() / PrePostProcessor.INPUT_WIDTH;
//                mImgScaleY = (float)mBitmap.getHeight() / PrePostProcessor.INPUT_HEIGHT;

                mImgScaleX = (float)mBitmap.getWidth();
                mImgScaleY = (float)mBitmap.getHeight();

                mIvScaleX = (mBitmap.getWidth() > mBitmap.getHeight() ? (float)mImageView.getWidth() / mBitmap.getWidth() : (float)mImageView.getHeight() / mBitmap.getHeight());
                mIvScaleY  = (mBitmap.getHeight() > mBitmap.getWidth() ? (float)mImageView.getHeight() / mBitmap.getHeight() : (float)mImageView.getWidth() / mBitmap.getWidth());

                mStartX = (mImageView.getWidth() - mIvScaleX * mBitmap.getWidth())/2;
                mStartY = (mImageView.getHeight() -  mIvScaleY * mBitmap.getHeight())/2;

                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });

        try {
//            mModule = PyTorchAndroid.loadModuleFromAsset(getAssets(), "d2go.pt");
            //model load
            mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "traced_net.ptl"));
            Log.v("Yolact", "loaded model.ptl");
            //class load
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open("classes.txt")));
            String line;
            List<String> classes = new ArrayList<>();
            while ((line = br.readLine()) != null) {
                classes.add(line);
            }
            Log.v("Yolact", "loaded class.txt");
            //class labels goes to PrePostprocessor
            PrePostProcessor.mClasses = new String[classes.size()];
            classes.toArray(PrePostProcessor.mClasses);
            Log.v("Yolact", "class>PrePostProcessor");
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }
    }

    // all activity related to Result, use requestCode(0,1), resultCode(Ok, not Ok), data
    // mBitmap > mimageview, which shows image to screen
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        Log.v("Yolact", "onActivityResult");
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_CANCELED) {
            switch (requestCode) {
                case 0:
                    if (resultCode == RESULT_OK && data != null) {
                        mBitmap = (Bitmap) data.getExtras().get("data");
                        Matrix matrix = new Matrix();
                        matrix.postRotate(90.0f);
                        mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                        mImageView.setImageBitmap(mBitmap);
                    }
                    break;
                case 1:
                    if (resultCode == RESULT_OK && data != null) {
                        Uri selectedImage = data.getData();
                        String[] filePathColumn = {MediaStore.Images.Media.DATA};
                        if (selectedImage != null) {
                            Cursor cursor = getContentResolver().query(selectedImage,
                                    filePathColumn, null, null, null);
                            if (cursor != null) {
                                cursor.moveToFirst();
                                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                                String picturePath = cursor.getString(columnIndex);
                                mBitmap = BitmapFactory.decodeFile(picturePath);
                                Matrix matrix = new Matrix();
                                matrix.postRotate(90.0f);
                                mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                                mImageView.setImageBitmap(mBitmap);
                                cursor.close();
                            }
                        }
                    }
                    break;
            }
        }
    }

    @Override
    public void run() {
        Log.v("Yolact", "run");
        // image resize to 550, 550
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(mBitmap, PrePostProcessor.INPUT_WIDTH, PrePostProcessor.INPUT_HEIGHT, true);
        Log.v("Yolact", "image resized");
        // image to 3, 550, 550 Tensor in buffer
        final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * resizedBitmap.getWidth() * resizedBitmap.getHeight());
        TensorImageUtils.bitmapToFloatBuffer(resizedBitmap, 0, 0, resizedBitmap.getWidth(), resizedBitmap.getHeight(), PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB, floatBuffer, 0);
        final Tensor inputTensor = Tensor.fromBlob(floatBuffer, new long[]{1, 3, resizedBitmap.getHeight(), resizedBitmap.getWidth()});
        Log.v("Yolact", "inputTensor : " + Arrays.toString(inputTensor.shape()));
        final long startTime = SystemClock.elapsedRealtime();
        IValue IvinputTensor = IValue.from(inputTensor);
        Log.v("Yolact", "IvinputTensor made: " + IvinputTensor.isNull());
//        IValue[] outputTuple = mModule.forward(IvinputTensor).toTuple();
        IValue Ivoutput = mModule.forward(IvinputTensor);
        Log.v("Yolact", "Ivoutput made");
        IValue[] outputTuple = Ivoutput.toTuple();
        Log.v("Yolact", "module forward");
        final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
        Log.d("Yolact", "inference time (ms): " + inferenceTime);
        Log.v("Yolact", "outputTuple len: " + outputTuple.length);
        Log.v("Yolact", "outputTuple[0]: " + Arrays.toString(outputTuple[0].toTensor().shape())); //loc
        Log.v("Yolact", "outputTuple[1]: " + Arrays.toString(outputTuple[1].toTensor().shape())); //conf
        Log.v("Yolact", "outputTuple[2]: " + Arrays.toString(outputTuple[2].toTensor().shape())); //mask
        Log.v("Yolact", "outputTuple[3]: " + Arrays.toString(outputTuple[3].toTensor().shape())); //prior
        Log.v("Yolact", "outputTuple[4]: " + Arrays.toString(outputTuple[4].toTensor().shape())); //proto

        float confthreshold = 0.4f;
        double iouthreshold = 0.5f;

        float[] locs = new float[]{};
        float[] conf = new float[]{};
        float[] mask = new float[]{};
        float[] priors = new float[]{};
        float[] proto = new float[]{};

        final Tensor locsTensor = outputTuple[0].toTensor();
        final Tensor confTensor = outputTuple[1].toTensor();
        final Tensor maskTensor = outputTuple[2].toTensor();
        final Tensor priorsTensor = outputTuple[3].toTensor();
        final Tensor protoTensor = outputTuple[4].toTensor();

        locs = locsTensor.getDataAsFloatArray();
        conf = confTensor.getDataAsFloatArray();
        mask = maskTensor.getDataAsFloatArray();
        priors = priorsTensor.getDataAsFloatArray();
        proto = protoTensor.getDataAsFloatArray();

        Log.v("Yolact", "locs len :" + locs.length);
        Log.v("Yolact", "conf len :" + conf.length);
        Log.v("Yolact", "mask len :" + mask.length);
        Log.v("Yolact", "priors len :" + priors.length);
        Log.v("Yolact", "proto len :" + proto.length);

        // to keep or not
        ArrayList<Integer> keep = new ArrayList<Integer>();
        float maxconf = 0.0f;
        for (int i = 0; i < 19248; i++) {
            maxconf = -Float.MAX_VALUE;
            for (int j = 0; j < 80; j++) {
                if (conf[i * 81 + j + 1] > maxconf) maxconf = conf[i * 81 + j + 1];
            }
            if (maxconf > confthreshold) keep.add(i);
        }
        Log.v("Yolact", "keep len :" + keep.size());
        if (keep.size()>0) {
            Log.v("Yolact", "keep : " + keep);

            ArrayList<Float> conf_keep = new ArrayList<Float>();
            ArrayList<Float> priors_keep = new ArrayList<Float>();
            ArrayList<Float> locs_keep = new ArrayList<Float>();
            ArrayList<Integer> class_keep = new ArrayList<Integer>();

            int target = -1;
            int maxclass = -1;
            for (int i = 0; i < keep.size(); i++) {
                maxconf = -Float.MAX_VALUE;
                target = keep.get(i);
                for (int j = 0; j < 80; j++) {
                    if (conf[target * 81 + j + 1] > maxconf) {
                        maxconf = conf[target * 81 + j + 1];
                        maxclass = j + 1;
                    }
                }
                conf_keep.add(maxconf);
                class_keep.add(maxclass);
                priors_keep.add(priors[4 * target + 0]);
                priors_keep.add(priors[4 * target + 1]);
                priors_keep.add(priors[4 * target + 2]);
                priors_keep.add(priors[4 * target + 3]);

                locs_keep.add(locs[4 * target + 0]);
                locs_keep.add(locs[4 * target + 1]);
                locs_keep.add(locs[4 * target + 2]);
                locs_keep.add(locs[4 * target + 3]);
            }
            Log.v("Yolact", "conf_keep len :" + conf_keep.size());
            Log.v("Yolact", "conf_keep:" + conf_keep);
            Log.v("Yolact", "class_keep len :" + class_keep.size());
            Log.v("Yolact", "class_keep:" + class_keep);
            Log.v("Yolact", "priors_keep len :" + priors_keep.size());
            Log.v("Yolact", "priors_keep:" + priors_keep);
            Log.v("Yolact", "loc_keep len :" + locs_keep.size());
            Log.v("Yolact", "loc_keep:" + locs_keep);

            float[] boxesData = new float[priors_keep.size()];
            double x1 = 0.0;
            double y1 = 0.0;
            double x2 = 0.0;
            double y2 = 0.0;
            for (int i = 0; i < keep.size(); i++) {
                x1 = priors_keep.get(4 * i) + locs_keep.get(4 * i) * 0.1 * priors_keep.get(4 * i + 2);
                y1 = priors_keep.get(4 * i + 1) + locs_keep.get(4 * i + 1) * 0.1 * priors_keep.get(4 * i + 3);
                x2 = priors_keep.get(4 * i + 2) * Math.exp(locs_keep.get(4 * i + 2) * 0.2);
                y2 = priors_keep.get(4 * i + 3) * Math.exp(locs_keep.get(4 * i + 3) * 0.2);

                x1 -= x2 / 2;
                y1 -= y2 / 2;
                x2 += x1;
                y2 += y1;

                boxesData[4 * i] = (float) x1;
                boxesData[4 * i + 1] = (float) y1;
                boxesData[4 * i + 2] = (float) x2;
                boxesData[4 * i + 3] = (float) y2;
            }
            Log.v("Yolact", "boxesData len :" + boxesData.length);
            Log.v("Yolact", "boxesData len :" + Arrays.toString(boxesData));

            int n = keep.size();
            Log.v("Yolact", "make n outputs :" + n);
            int count = 0;
            ArrayList<Float> outputs = new ArrayList<Float>();
            for (int i = 0; i < n; i++) {
                outputs.add(boxesData[4 * i + 0]);
                outputs.add(boxesData[4 * i + 1]);
                outputs.add(boxesData[4 * i + 2]);
                outputs.add(boxesData[4 * i + 3]);
                outputs.add(conf_keep.get(i));
                outputs.add((float) class_keep.get(i) - 1);
                outputs.add((float) keep.get(i));
                count++;
            }
            Log.v("Yolact", "outputs len: " + outputs.size());
            ArrayList<Float> nmsoutputs = PrePostProcessor.nonMaxSuppresion(count, outputs, iouthreshold);
            Log.v("Yolact", "nmsoutputs len :" + nmsoutputs.size());
            float [] arrnmsoutputs = new float[nmsoutputs.size()];
            for (int i=0; i < nmsoutputs.size(); i++) {
                arrnmsoutputs[i] = nmsoutputs.get(i);
            }
            Log.v("Yolact", "nmsoutputs len :" + arrnmsoutputs.length);
            count = arrnmsoutputs.length/7;
            Log.v("Yolact", "draw n box :" + count);
            final ArrayList<Result> results = PrePostProcessor.outputsToPredictions(count, arrnmsoutputs, mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY);
            Log.v("Yolact", "postprocessed");

            //draw mask
            ArrayList<Integer> colorlist = new ArrayList<Integer>();
            int[] rs = {40,80,120,160,200};
            int[] gs = {50,100,150,200};
            int[] bs = {50,100,150,200};
            for (int r : rs) {
                for (int g : gs) {
                    for (int b : bs) {
                        colorlist.add(Color.argb(120,r,g,b));
                    }
                }
            }
            Log.v("Yolact", "colorspace made");

            int[] pixels =  new int[138*138];
            for (int i = 0; i < count; i++) {
                for (int j = 0; j < 138; j++) {
                    for (int k = 0; k < 138; k++) {
                        float sum = 0.0f;
                        for (int l = 0; l < 32; l++) {
                            sum += proto[(j * 138 + k) * 32 + l] * mask[nmsoutputs.get(i*7 + 6).intValue()*32 + l];
                        }
                        sum = (float) (1/ (1+ Math.exp(-sum)));

                        if (sum > 0.5) {
                            pixels[j*138 + k] = colorlist.get(nmsoutputs.get(i * 7 + 5).intValue());
                        }
                    }
                }
            }

            Log.v("Yolact", "mask created");
            Bitmap maskbitmap = Bitmap.createBitmap(pixels,138,138, Bitmap.Config.ARGB_8888);
            Log.v("Yolact", "maskbitmap created");
            double newWidth = (mImgScaleX*mIvScaleX);
            double newHeight = (mImgScaleY*mIvScaleY);
            Bitmap resizedmask = Bitmap.createScaledBitmap(maskbitmap, (int) newWidth, (int) newHeight,true);
            Log.v("Yolact", "resizedmask created");

            runOnUiThread(() -> {
                mButtonDetect.setEnabled(true);
                mButtonDetect.setText(getString(R.string.detect));
                mProgressBar.setVisibility(ProgressBar.INVISIBLE);
                mResultView.setResults(results);
                mResultView.setOverlay(resizedmask);
                mResultView.invalidate();
                mResultView.setVisibility(View.VISIBLE);
            });
        }
        Log.v("Yolact", "draw!");
    }
}
