package org.pytorch.demo.objectdetection;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.Log;
import android.view.TextureView;
import android.view.ViewStub;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Map;


public class ObjectDetectionActivity extends AbstractCameraXActivity<ObjectDetectionActivity.AnalysisResult> {
    private Module mModule = null;
    private ResultView mResultView;

    static class AnalysisResult {
        private final ArrayList<Result> mResults;

        public AnalysisResult(ArrayList<Result> results) {
            mResults = results;
        }
    }

    @Override
    protected int getContentViewLayoutId() {
        return R.layout.activity_object_detection;
    }

    @Override
    protected TextureView getCameraPreviewTextureView() {
        mResultView = findViewById(R.id.resultView);
        return ((ViewStub) findViewById(R.id.object_detection_texture_view_stub))
                .inflate()
                .findViewById(R.id.object_detection_texture_view);
    }

    @Override
    protected void applyToUiAnalyzeImageResult(AnalysisResult result) {
        mResultView.setResults(result.mResults);
        mResultView.invalidate();
    }

    private Bitmap imgToBitmap(Image image) {
//        Log.v("YolactActivity","imgToBitmap");
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    public static String assetFilePath(Context context, String assetName) throws IOException {
//        Log.v("YolactActivity","assetFilePath");
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
    @WorkerThread
    @Nullable
    protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
//        Log.v("YolactActivity","analyzeImage");
        if (mModule == null) {
//            Log.v("YolactActivity","loading module");
//            mModule = PyTorchAndroid.loadModuleFromAsset(getAssets(), "traced_net.ptl");
            try {
                mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "traced_net.ptl"));
            } catch (IOException e) {
                e.printStackTrace();
            }
//            Log.v("YolactActivity","loaded module");
        }
        Bitmap bitmap = imgToBitmap(image.getImage());
        Matrix matrix = new Matrix();
        matrix.postRotate(90.0f);
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, PrePostProcessor.INPUT_WIDTH, PrePostProcessor.INPUT_HEIGHT, true);
        final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * resizedBitmap.getWidth() * resizedBitmap.getHeight());
        TensorImageUtils.bitmapToFloatBuffer(resizedBitmap, 0,0,resizedBitmap.getWidth(),resizedBitmap.getHeight(), PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB, floatBuffer, 0);
        final Tensor inputTensor =  Tensor.fromBlob(floatBuffer, new long[] {1,3, resizedBitmap.getHeight(), resizedBitmap.getWidth()});
//        Log.v("YolactActivity","inputTensor created");
        IValue IvinputTensor = IValue.from(inputTensor);
        IValue[] outputTuple = mModule.forward(IvinputTensor).toTuple();
//        Log.v("YolactActivity","outputTuple created len: "+ outputTuple.length);

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

        ArrayList<Integer> keep = new ArrayList<Integer>();
        float maxconf = 0.0f;
        for (int i = 0; i < 19248; i++) {
            maxconf = -Float.MAX_VALUE;
            for (int j = 0; j < 80; j++) {
                if (conf[i * 81 + j + 1] > maxconf) maxconf = conf[i * 81 + j + 1];
            }
            if (maxconf > confthreshold) keep.add(i);
        }

        if (keep.size() > 0) {
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

            int n = keep.size();
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
            ArrayList<Float> nmsoutputs = PrePostProcessor.nonMaxSuppresion(count, outputs, iouthreshold);
            float [] arrnmsoutputs = new float[nmsoutputs.size()];
            for (int i=0; i < nmsoutputs.size(); i++) {
                arrnmsoutputs[i] = nmsoutputs.get(i);
            }
            count = arrnmsoutputs.length/7;

            float imgScaleX = (float) bitmap.getWidth();
            float imgScaleY = (float) bitmap.getHeight();
            float ivScaleX = (float) mResultView.getWidth() / bitmap.getWidth();
            float ivScaleY = (float) mResultView.getHeight() / bitmap.getHeight();

            final ArrayList<Result> results = PrePostProcessor.outputsToPredictions(count, arrnmsoutputs, imgScaleX, imgScaleY, ivScaleX, ivScaleY, 0, 0);

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

            Bitmap maskbitmap = Bitmap.createBitmap(pixels,138,138, Bitmap.Config.ARGB_8888);
            double newWidth = (imgScaleX*ivScaleX);
            double newHeight = (imgScaleY*ivScaleY);
            Bitmap resizedmask = Bitmap.createScaledBitmap(maskbitmap, (int) newWidth, (int) newHeight,true);

            mResultView.setOverlay(resizedmask);
            return new AnalysisResult(results);
        }
        return null;
    }
}
