package mobile_app.android_ai_integration;

import android.graphics.Bitmap;
import android.graphics.Color;

final class HandwritingEnhancer {

    enum Mode {
        NATURAL,
        GRAYSCALE,
        BLACK_AND_WHITE
    }

    private HandwritingEnhancer() {
    }

    static Bitmap enhance(Bitmap source, Mode mode, int strengthPercent) {
        int width = source.getWidth();
        int height = source.getHeight();
        int size = width * height;
        int[] sourcePixels = new int[size];
        int[] gray = new int[size];
        int[] background = new int[size];
        int[] outputPixels = new int[size];

        source.getPixels(sourcePixels, 0, width, 0, 0, width, height);
        for (int index = 0; index < size; index++) {
            int color = sourcePixels[index];
            int red = Color.red(color);
            int green = Color.green(color);
            int blue = Color.blue(color);
            gray[index] = (red * 77 + green * 150 + blue * 29) >> 8;
        }

        int radius = Math.max(8, Math.min(48, Math.min(width, height) / 28));
        boxBlur(gray, background, width, height, radius);

        float strength = Math.max(0, Math.min(100, strengthPercent)) / 100f;
        float inkBoost = 1.05f + (1.45f * strength);
        int binaryThreshold = Math.round(188f + (43f * strength));

        for (int index = 0; index < size; index++) {
            int localBackground = Math.max(1, background[index]);
            int normalized = clamp(Math.round((gray[index] * 255f) / localBackground));
            int enhancedLuma = clamp(Math.round(255f - ((255 - normalized) * inkBoost)));

            if (mode == Mode.BLACK_AND_WHITE) {
                int value = normalized < binaryThreshold ? 0 : 255;
                outputPixels[index] = Color.rgb(value, value, value);
            } else if (mode == Mode.GRAYSCALE) {
                outputPixels[index] = Color.rgb(enhancedLuma, enhancedLuma, enhancedLuma);
            } else {
                int original = sourcePixels[index];
                int originalLuma = Math.max(1, gray[index]);
                float chromaAmount = 0.58f;
                int red = clamp(Math.round(enhancedLuma
                        + ((Color.red(original) - originalLuma) * chromaAmount)));
                int green = clamp(Math.round(enhancedLuma
                        + ((Color.green(original) - originalLuma) * chromaAmount)));
                int blue = clamp(Math.round(enhancedLuma
                        + ((Color.blue(original) - originalLuma) * chromaAmount)));
                outputPixels[index] = Color.rgb(red, green, blue);
            }
        }

        Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        result.setPixels(outputPixels, 0, width, 0, 0, width, height);
        return result;
    }

    private static void boxBlur(int[] input, int[] output, int width, int height, int radius) {
        int[] horizontal = new int[input.length];

        for (int y = 0; y < height; y++) {
            int rowStart = y * width;
            long sum = 0;
            int firstRight = Math.min(width - 1, radius);
            for (int x = 0; x <= firstRight; x++) {
                sum += input[rowStart + x];
            }

            for (int x = 0; x < width; x++) {
                int left = Math.max(0, x - radius);
                int right = Math.min(width - 1, x + radius);
                horizontal[rowStart + x] = (int) (sum / (right - left + 1));

                int leaving = x - radius;
                int entering = x + radius + 1;
                if (leaving >= 0) {
                    sum -= input[rowStart + leaving];
                }
                if (entering < width) {
                    sum += input[rowStart + entering];
                }
            }
        }

        for (int x = 0; x < width; x++) {
            long sum = 0;
            int firstBottom = Math.min(height - 1, radius);
            for (int y = 0; y <= firstBottom; y++) {
                sum += horizontal[(y * width) + x];
            }

            for (int y = 0; y < height; y++) {
                int top = Math.max(0, y - radius);
                int bottom = Math.min(height - 1, y + radius);
                output[(y * width) + x] = (int) (sum / (bottom - top + 1));

                int leaving = y - radius;
                int entering = y + radius + 1;
                if (leaving >= 0) {
                    sum -= horizontal[(leaving * width) + x];
                }
                if (entering < height) {
                    sum += horizontal[(entering * width) + x];
                }
            }
        }
    }

    private static int clamp(int value) {
        return Math.max(0, Math.min(255, value));
    }
}
