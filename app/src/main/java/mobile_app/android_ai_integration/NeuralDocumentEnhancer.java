package mobile_app.android_ai_integration;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * Runs the bundled pretrained document-binarization network in 224 px tiles.
 *
 * <p>The network predicts the probability that each pixel belongs to clean paper. Images are
 * downscaled to a bounded inference canvas and the resulting mask is upscaled before rendering,
 * which keeps memory and latency predictable on mobile devices.</p>
 */
final class NeuralDocumentEnhancer implements AutoCloseable {

    static final int TILE_SIZE = 224;
    static final int MAX_INFERENCE_DIMENSION = 896;
    private static final String MODEL_ASSET = "models/neural_clean.tflite";

    private final Context appContext;
    private Interpreter interpreter;

    NeuralDocumentEnhancer(Context context) {
        appContext = context.getApplicationContext();
    }

    synchronized Bitmap enhance(
            Bitmap source,
            HandwritingEnhancer.Mode mode,
            int strengthPercent
    ) throws IOException {
        ensureInterpreter();

        float scale = Math.min(
                1f,
                MAX_INFERENCE_DIMENSION / (float) Math.max(source.getWidth(), source.getHeight())
        );
        int inferenceWidth = Math.max(1, Math.round(source.getWidth() * scale));
        int inferenceHeight = Math.max(1, Math.round(source.getHeight() * scale));
        Bitmap inferenceBitmap = scale < 1f
                ? Bitmap.createScaledBitmap(source, inferenceWidth, inferenceHeight, true)
                : source;

        int[] inputPixels = new int[inferenceWidth * inferenceHeight];
        int[] maskPixels = new int[inputPixels.length];
        inferenceBitmap.getPixels(inputPixels, 0, inferenceWidth, 0, 0, inferenceWidth, inferenceHeight);

        ByteBuffer input = ByteBuffer.allocateDirect(TILE_SIZE * TILE_SIZE * 3 * Float.BYTES)
                .order(ByteOrder.nativeOrder());
        float[][][][] output = new float[1][TILE_SIZE][TILE_SIZE][1];

        for (int top = 0; top < inferenceHeight; top += TILE_SIZE) {
            for (int left = 0; left < inferenceWidth; left += TILE_SIZE) {
                fillTile(input, inputPixels, inferenceWidth, inferenceHeight, left, top);
                interpreter.run(input, output);
                writeMaskTile(
                        output,
                        maskPixels,
                        inferenceWidth,
                        inferenceHeight,
                        left,
                        top,
                        strengthPercent
                );
            }
        }

        Bitmap mask = Bitmap.createBitmap(
                maskPixels,
                inferenceWidth,
                inferenceHeight,
                Bitmap.Config.ARGB_8888
        );
        Bitmap fullSizeMask = inferenceWidth == source.getWidth()
                && inferenceHeight == source.getHeight()
                ? mask
                : Bitmap.createScaledBitmap(mask, source.getWidth(), source.getHeight(), true);

        Bitmap rendered = render(source, fullSizeMask, mode, strengthPercent);

        if (fullSizeMask != mask) {
            fullSizeMask.recycle();
        }
        mask.recycle();
        if (inferenceBitmap != source) {
            inferenceBitmap.recycle();
        }
        return rendered;
    }

    private void fillTile(
            ByteBuffer input,
            int[] pixels,
            int width,
            int height,
            int left,
            int top
    ) {
        input.rewind();
        for (int y = 0; y < TILE_SIZE; y++) {
            int sourceY = top + y;
            for (int x = 0; x < TILE_SIZE; x++) {
                int sourceX = left + x;
                int color = sourceX < width && sourceY < height
                        ? pixels[(sourceY * width) + sourceX]
                        : Color.WHITE;
                input.putFloat(Color.red(color));
                input.putFloat(Color.green(color));
                input.putFloat(Color.blue(color));
            }
        }
        input.rewind();
    }

    private void writeMaskTile(
            float[][][][] output,
            int[] maskPixels,
            int width,
            int height,
            int left,
            int top,
            int strengthPercent
    ) {
        float strength = clamp01(strengthPercent / 100f);
        float contrast = 1.4f + (1.8f * strength);
        for (int y = 0; y < TILE_SIZE && top + y < height; y++) {
            int row = (top + y) * width;
            for (int x = 0; x < TILE_SIZE && left + x < width; x++) {
                float probability = sigmoid(output[0][y][x][0]);
                float cleaned = clamp01(((probability - 0.5f) * contrast) + 0.5f);
                int value = Math.round(cleaned * 255f);
                maskPixels[row + left + x] = Color.rgb(value, value, value);
            }
        }
    }

    private Bitmap render(
            Bitmap source,
            Bitmap mask,
            HandwritingEnhancer.Mode mode,
            int strengthPercent
    ) {
        int width = source.getWidth();
        int height = source.getHeight();
        int size = width * height;
        int[] sourcePixels = new int[size];
        int[] maskPixels = new int[size];
        int[] outputPixels = new int[size];
        source.getPixels(sourcePixels, 0, width, 0, 0, width, height);
        mask.getPixels(maskPixels, 0, width, 0, 0, width, height);

        float strength = clamp01(strengthPercent / 100f);
        float blend = 0.55f + (0.4f * strength);
        int threshold = Math.round(142f + (34f * strength));

        for (int index = 0; index < size; index++) {
            int original = sourcePixels[index];
            int originalLuma = (
                    Color.red(original) * 77
                            + Color.green(original) * 150
                            + Color.blue(original) * 29
            ) >> 8;
            int maskLuma = Color.red(maskPixels[index]);

            if (mode == HandwritingEnhancer.Mode.BLACK_AND_WHITE) {
                int value = maskLuma < threshold ? 0 : 255;
                outputPixels[index] = Color.rgb(value, value, value);
                continue;
            }

            int cleanLuma = clamp(Math.round(originalLuma + ((maskLuma - originalLuma) * blend)));
            if (mode == HandwritingEnhancer.Mode.GRAYSCALE) {
                outputPixels[index] = Color.rgb(cleanLuma, cleanLuma, cleanLuma);
                continue;
            }

            float chromaAmount = 0.38f;
            int red = clamp(Math.round(
                    cleanLuma + ((Color.red(original) - originalLuma) * chromaAmount)
            ));
            int green = clamp(Math.round(
                    cleanLuma + ((Color.green(original) - originalLuma) * chromaAmount)
            ));
            int blue = clamp(Math.round(
                    cleanLuma + ((Color.blue(original) - originalLuma) * chromaAmount)
            ));
            outputPixels[index] = Color.rgb(red, green, blue);
        }

        Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        result.setPixels(outputPixels, 0, width, 0, 0, width, height);
        return result;
    }

    private void ensureInterpreter() throws IOException {
        if (interpreter != null) {
            return;
        }
        Interpreter.Options options = new Interpreter.Options()
                .setNumThreads(Math.max(2, Math.min(4, Runtime.getRuntime().availableProcessors())))
                .setUseXNNPACK(true);
        interpreter = new Interpreter(loadModel(), options);
    }

    private MappedByteBuffer loadModel() throws IOException {
        try (AssetFileDescriptor descriptor = appContext.getAssets().openFd(MODEL_ASSET);
             FileInputStream stream = new FileInputStream(descriptor.getFileDescriptor());
             FileChannel channel = stream.getChannel()) {
            return channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    descriptor.getStartOffset(),
                    descriptor.getDeclaredLength()
            );
        }
    }

    static float sigmoid(float value) {
        if (value >= 0f) {
            double exponent = Math.exp(-value);
            return (float) (1d / (1d + exponent));
        }
        double exponent = Math.exp(value);
        return (float) (exponent / (1d + exponent));
    }

    private static float clamp01(float value) {
        return Math.max(0f, Math.min(1f, value));
    }

    private static int clamp(int value) {
        return Math.max(0, Math.min(255, value));
    }

    @Override
    public synchronized void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
    }
}