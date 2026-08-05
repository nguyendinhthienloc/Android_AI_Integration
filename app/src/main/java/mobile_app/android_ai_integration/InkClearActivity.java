package mobile_app.android_ai_integration;

import android.app.Activity;
import android.content.ClipData;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.ImageDecoder;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.pdf.PdfDocument;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.RadioGroup;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.ComponentActivity;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.IntentSenderRequest;
import androidx.activity.result.contract.ActivityResultContracts;

import com.google.mlkit.vision.documentscanner.GmsDocumentScanner;
import com.google.mlkit.vision.documentscanner.GmsDocumentScannerOptions;
import com.google.mlkit.vision.documentscanner.GmsDocumentScanning;
import com.google.mlkit.vision.documentscanner.GmsDocumentScanningResult;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class InkClearActivity extends ComponentActivity {

    private static final int MAX_IMAGE_DIMENSION = 1800;
    private static final String STATE_IMAGE_URI = "image_uri";
    private static final String STATE_MODE = "mode";
    private static final String STATE_STRENGTH = "strength";

    private final ExecutorService processingExecutor = Executors.newSingleThreadExecutor();
    private int processingGeneration = 0;
    private int ocrGeneration = 0;

    private ImageView documentPreview;
    private View emptyState;
    private ProgressBar processingProgress;
    private ProgressBar ocrProgress;
    private Button originalButton;
    private Button enhancedButton;
    private Button saveImageButton;
    private Button savePdfButton;
    private Button recognizeTextButton;
    private SeekBar strengthSlider;
    private TextView strengthValue;
    private TextView processingMethodBadge;

    private Uri selectedImageUri;
    private Bitmap originalBitmap;
    private Bitmap enhancedBitmap;
    private HandwritingEnhancer.Mode selectedMode = HandwritingEnhancer.Mode.NATURAL;
    private boolean showingOriginal = false;
    private boolean neuralFallbackAnnounced = false;
    private GmsDocumentScanner documentScanner;
    private NeuralDocumentEnhancer neuralEnhancer;
    private OfflineOcrEngine ocrEngine;

    private final ActivityResultLauncher<IntentSenderRequest> documentScannerLauncher =
            registerForActivityResult(
                    new ActivityResultContracts.StartIntentSenderForResult(),
                    result -> {
                        if (result.getResultCode() != Activity.RESULT_OK || result.getData() == null) {
                            return;
                        }

                        GmsDocumentScanningResult scanResult =
                                GmsDocumentScanningResult.fromActivityResultIntent(result.getData());
                        if (scanResult == null
                                || scanResult.getPages() == null
                                || scanResult.getPages().isEmpty()) {
                            Toast.makeText(
                                    this,
                                    R.string.scanner_unavailable,
                                    Toast.LENGTH_LONG
                            ).show();
                            return;
                        }

                        selectedImageUri = scanResult.getPages().get(0).getImageUri();
                        loadImage(selectedImageUri);
                    }
            );

    private final ActivityResultLauncher<String[]> imagePicker = registerForActivityResult(
            new ActivityResultContracts.OpenDocument(),
            uri -> {
                if (uri == null) {
                    return;
                }
                selectedImageUri = uri;
                try {
                    getContentResolver().takePersistableUriPermission(
                            uri,
                            Intent.FLAG_GRANT_READ_URI_PERMISSION
                    );
                } catch (SecurityException ignored) {
                    // Some document providers grant access only for the current activity.
                }
                loadImage(uri);
            }
    );

    private final ActivityResultLauncher<String> imageSaver = registerForActivityResult(
            new ActivityResultContracts.CreateDocument("image/jpeg"),
            uri -> {
                if (uri != null && enhancedBitmap != null) {
                    saveEnhancedImage(uri);
                }
            }
    );

    private final ActivityResultLauncher<String> pdfSaver = registerForActivityResult(
            new ActivityResultContracts.CreateDocument("application/pdf"),
            uri -> {
                if (uri != null && enhancedBitmap != null) {
                    saveEnhancedPdf(uri);
                }
            }
    );

    private final ActivityResultLauncher<Intent> ocrResultLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == Activity.RESULT_OK) {
                    recognizeText();
                }
            }
    );

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.ink_clear_activity);

        neuralEnhancer = new NeuralDocumentEnhancer(this);
        ocrEngine = new OfflineOcrEngine(this);

        bindViews();
        configureDocumentScanner();
        configureControls();

        if (savedInstanceState != null) {
            selectedMode = HandwritingEnhancer.Mode.valueOf(
                    savedInstanceState.getString(STATE_MODE, HandwritingEnhancer.Mode.NATURAL.name())
            );
            int strength = savedInstanceState.getInt(STATE_STRENGTH, 60);
            strengthSlider.setProgress(strength);
            setCheckedMode(selectedMode);

            String uriValue = savedInstanceState.getString(STATE_IMAGE_URI);
            if (uriValue != null) {
                selectedImageUri = Uri.parse(uriValue);
                loadImage(selectedImageUri);
            }
        } else if (getIntent() != null && getIntent().getData() != null) {
            selectedImageUri = getIntent().getData();
            loadImage(selectedImageUri);
        }
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        if (intent != null && intent.getData() != null) {
            selectedImageUri = intent.getData();
            loadImage(selectedImageUri);
        }
    }

    private void bindViews() {
        documentPreview = findViewById(R.id.document_preview);
        emptyState = findViewById(R.id.empty_state);
        processingProgress = findViewById(R.id.processing_progress);
        ocrProgress = findViewById(R.id.ocr_progress);
        originalButton = findViewById(R.id.show_original_button);
        enhancedButton = findViewById(R.id.show_enhanced_button);
        saveImageButton = findViewById(R.id.save_image_button);
        savePdfButton = findViewById(R.id.save_pdf_button);
        recognizeTextButton = findViewById(R.id.recognize_text_button);
        strengthSlider = findViewById(R.id.strength_slider);
        strengthValue = findViewById(R.id.strength_value);
        processingMethodBadge = findViewById(R.id.processing_method_badge);
    }

    private void configureControls() {
        findViewById(R.id.back_button).setOnClickListener(view -> finish());
        findViewById(R.id.ai_scan_button).setOnClickListener(view -> launchDocumentScanner());
        findViewById(R.id.choose_image_button).setOnClickListener(
                view -> imagePicker.launch(new String[]{"image/*"})
        );

        originalButton.setOnClickListener(view -> {
            showingOriginal = true;
            renderPreview();
        });
        enhancedButton.setOnClickListener(view -> {
            showingOriginal = false;
            renderPreview();
        });

        RadioGroup modeGroup = findViewById(R.id.mode_group);
        modeGroup.setOnCheckedChangeListener((group, checkedId) -> {
            if (checkedId == R.id.mode_grayscale) {
                selectedMode = HandwritingEnhancer.Mode.GRAYSCALE;
            } else if (checkedId == R.id.mode_black_white) {
                selectedMode = HandwritingEnhancer.Mode.BLACK_AND_WHITE;
            } else {
                selectedMode = HandwritingEnhancer.Mode.NATURAL;
            }
            processOriginal();
        });

        strengthSlider.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                strengthValue.setText(getString(R.string.percent_value, progress));
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                processOriginal();
            }
        });

        saveImageButton.setOnClickListener(view -> imageSaver.launch("InkClear-enhanced.jpg"));
        savePdfButton.setOnClickListener(view -> pdfSaver.launch("InkClear-enhanced.pdf"));
        recognizeTextButton.setOnClickListener(view -> recognizeText());
        updateControlState(false);
    }

    private void configureDocumentScanner() {
        try {
            GmsDocumentScannerOptions options = new GmsDocumentScannerOptions.Builder()
                    .setGalleryImportAllowed(true)
                    .setPageLimit(1)
                    .setResultFormats(GmsDocumentScannerOptions.RESULT_FORMAT_JPEG)
                    .setScannerMode(GmsDocumentScannerOptions.SCANNER_MODE_FULL)
                    .build();
            documentScanner = GmsDocumentScanning.getClient(options);
        } catch (Exception exception) {
            documentScanner = null;
        }
    }

    private void launchDocumentScanner() {
        if (documentScanner == null) {
            Toast.makeText(
                    this,
                    R.string.scanner_unavailable,
                    Toast.LENGTH_LONG
            ).show();
            return;
        }
        documentScanner.getStartScanIntent(this)
                .addOnSuccessListener(intentSender -> documentScannerLauncher.launch(
                        new IntentSenderRequest.Builder(intentSender).build()
                ))
                .addOnFailureListener(exception -> Toast.makeText(
                        this,
                        R.string.scanner_unavailable,
                        Toast.LENGTH_LONG
                ).show());
    }

    private void loadImage(Uri uri) {
        cancelOcr();
        int requestGeneration = ++processingGeneration;
        updateControlState(false);
        setBusy(true);
        processingExecutor.execute(() -> {
            try {
                Bitmap bitmap = decodeBitmap(uri);
                runOnUiThread(() -> {
                    if (requestGeneration != processingGeneration || isDestroyed()) {
                        bitmap.recycle();
                        return;
                    }
                    recycleBitmap(originalBitmap);
                    originalBitmap = bitmap;
                    processOriginal();
                });
            } catch (Exception exception) {
                runOnUiThread(() -> {
                    if (requestGeneration == processingGeneration) {
                        setBusy(false);
                        Toast.makeText(this, R.string.image_load_failed, Toast.LENGTH_LONG).show();
                    }
                });
            }
        });
    }

    private Bitmap decodeBitmap(Uri uri) throws IOException {
        Bitmap decoded = null;
        try {
            BitmapFactory.Options bounds = new BitmapFactory.Options();
            bounds.inJustDecodeBounds = true;
            try (InputStream stream = getContentResolver().openInputStream(uri)) {
                BitmapFactory.decodeStream(stream, null, bounds);
            }

            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            options.inSampleSize = calculateSampleSize(bounds.outWidth, bounds.outHeight);
            try (InputStream stream = getContentResolver().openInputStream(uri)) {
                decoded = BitmapFactory.decodeStream(stream, null, options);
            }
        } catch (Exception ignored) {
        }

        if (decoded == null && "file".equals(uri.getScheme())) {
            BitmapFactory.Options bounds = new BitmapFactory.Options();
            bounds.inJustDecodeBounds = true;
            BitmapFactory.decodeFile(uri.getPath(), bounds);
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            options.inSampleSize = calculateSampleSize(bounds.outWidth, bounds.outHeight);
            decoded = BitmapFactory.decodeFile(uri.getPath(), options);
        }

        if (decoded == null) {
            throw new IOException("Failed to decode bitmap from URI: " + uri);
        }
        return applyExifRotation(uri, decoded);
    }

    private int calculateSampleSize(int width, int height) {
        int sampleSize = 1;
        while (Math.max(width / sampleSize, height / sampleSize) > MAX_IMAGE_DIMENSION) {
            sampleSize *= 2;
        }
        return sampleSize;
    }

    private Bitmap applyExifRotation(Uri uri, Bitmap bitmap) {
        try (InputStream stream = getContentResolver().openInputStream(uri)) {
            ExifInterface exif = new ExifInterface(stream);
            int orientation = exif.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
            );
            float degrees;
            if (orientation == ExifInterface.ORIENTATION_ROTATE_90) {
                degrees = 90f;
            } else if (orientation == ExifInterface.ORIENTATION_ROTATE_180) {
                degrees = 180f;
            } else if (orientation == ExifInterface.ORIENTATION_ROTATE_270) {
                degrees = 270f;
            } else {
                return bitmap;
            }

            Matrix matrix = new Matrix();
            matrix.postRotate(degrees);
            Bitmap rotated = Bitmap.createBitmap(
                    bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true
            );
            if (rotated != bitmap) {
                bitmap.recycle();
            }
            return rotated;
        } catch (IOException exception) {
            return bitmap;
        }
    }

    private void processOriginal() {
        if (originalBitmap == null) {
            return;
        }

        cancelOcr();
        Bitmap source = originalBitmap;
        HandwritingEnhancer.Mode mode = selectedMode;
        int strength = strengthSlider.getProgress();
        int requestGeneration = ++processingGeneration;
        updateControlState(false);
        setBusy(true);

        processingExecutor.execute(() -> {
            Bitmap result;
            boolean usedNeuralModel = true;
            try {
                result = neuralEnhancer.enhance(source, mode, strength);
            } catch (Exception neuralError) {
                usedNeuralModel = false;
                try {
                    result = HandwritingEnhancer.enhance(source, mode, strength);
                } catch (Exception fallbackError) {
                    runOnUiThread(() -> {
                        if (requestGeneration == processingGeneration) {
                            setBusy(false);
                            Toast.makeText(
                                    this,
                                    R.string.image_process_failed,
                                    Toast.LENGTH_LONG
                            ).show();
                        }
                    });
                    return;
                }
            }

            Bitmap finalResult = result;
            boolean finalUsedNeuralModel = usedNeuralModel;
            runOnUiThread(() -> {
                if (requestGeneration != processingGeneration || isDestroyed()) {
                    finalResult.recycle();
                    return;
                }
                recycleBitmap(enhancedBitmap);
                enhancedBitmap = finalResult;
                showingOriginal = false;
                processingMethodBadge.setText(
                        finalUsedNeuralModel
                                ? R.string.neural_clean_badge
                                : R.string.classical_fallback_badge
                );
                processingMethodBadge.setVisibility(View.VISIBLE);
                setBusy(false);
                updateControlState(true);
                renderPreview();
                if (getIntent() != null && getIntent().getBooleanExtra("auto_ocr", false)) {
                    getIntent().removeExtra("auto_ocr");
                    recognizeText();
                }

                if (!finalUsedNeuralModel && !neuralFallbackAnnounced) {
                    neuralFallbackAnnounced = true;
                    Toast.makeText(
                            this,
                            R.string.neural_fallback_message,
                            Toast.LENGTH_LONG
                    ).show();
                }
            });
        });
    }

    private void recognizeText() {
        if (enhancedBitmap == null) {
            return;
        }

        cancelOcr();
        int requestGeneration = ++ocrGeneration;
        Bitmap ocrBitmap = enhancedBitmap.copy(Bitmap.Config.ARGB_8888, false);
        setOcrBusy(true);

        ocrEngine.recognize(ocrBitmap, new OfflineOcrEngine.Callback() {
            @Override
            public void onSuccess(OfflineOcrEngine.Result result) {
                ocrBitmap.recycle();
                if (requestGeneration != ocrGeneration || isDestroyed()) {
                    return;
                }
                setOcrBusy(false);
                openOcrResult(result);
            }

            @Override
            public void onFailure(Throwable error) {
                ocrBitmap.recycle();
                if (requestGeneration != ocrGeneration || isDestroyed()) {
                    return;
                }
                setOcrBusy(false);
                Toast.makeText(
                        InkClearActivity.this,
                        R.string.ocr_failed,
                        Toast.LENGTH_LONG
                ).show();
            }
        });
    }

    private void openOcrResult(OfflineOcrEngine.Result result) {
        Intent intent = new Intent(this, OcrResultActivity.class)
                .putExtra(OcrResultActivity.EXTRA_TEXT, result.getText())
                .putExtra(OcrResultActivity.EXTRA_LINE_COUNT, result.getLineCount())
                .putExtra(OcrResultActivity.EXTRA_DURATION_MS, result.getDurationMs())
                .putExtra(
                        OcrResultActivity.EXTRA_AVERAGE_CONFIDENCE,
                        result.getAverageConfidence()
                );

        if (selectedImageUri != null) {
            intent.setData(selectedImageUri);
            intent.setClipData(ClipData.newUri(
                    getContentResolver(),
                    "InkClear source",
                    selectedImageUri
            ));
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        }
        ocrResultLauncher.launch(intent);
    }

    private void cancelOcr() {
        ocrGeneration++;
        if (ocrEngine != null) {
            ocrEngine.cancel();
        }
        if (ocrProgress != null) {
            setOcrBusy(false);
        }
    }

    private void renderPreview() {
        Bitmap displayed = showingOriginal ? originalBitmap : enhancedBitmap;
        if (displayed == null) {
            return;
        }
        emptyState.setVisibility(View.GONE);
        documentPreview.setVisibility(View.VISIBLE);
        documentPreview.setImageBitmap(displayed);
        originalButton.setAlpha(showingOriginal ? 1f : 0.65f);
        enhancedButton.setAlpha(showingOriginal ? 0.65f : 1f);
    }

    private void saveEnhancedImage(Uri destination) {
        Bitmap bitmap = enhancedBitmap;
        processingExecutor.execute(() -> {
            try (OutputStream output = getContentResolver().openOutputStream(destination)) {
                if (output == null || !bitmap.compress(Bitmap.CompressFormat.JPEG, 95, output)) {
                    throw new IOException("Image compression failed");
                }
                runOnUiThread(() -> Toast.makeText(
                        this,
                        R.string.image_saved,
                        Toast.LENGTH_SHORT
                ).show());
            } catch (IOException exception) {
                runOnUiThread(() -> Toast.makeText(
                        this,
                        R.string.save_failed,
                        Toast.LENGTH_LONG
                ).show());
            }
        });
    }

    private void saveEnhancedPdf(Uri destination) {
        Bitmap bitmap = enhancedBitmap;
        processingExecutor.execute(() -> {
            PdfDocument document = new PdfDocument();
            try (OutputStream output = getContentResolver().openOutputStream(destination)) {
                if (output == null) {
                    throw new IOException("Unable to open destination");
                }

                PdfDocument.PageInfo pageInfo = new PdfDocument.PageInfo.Builder(
                        bitmap.getWidth(), bitmap.getHeight(), 1
                ).create();
                PdfDocument.Page page = document.startPage(pageInfo);
                Canvas canvas = page.getCanvas();
                canvas.drawColor(android.graphics.Color.WHITE);
                canvas.drawBitmap(bitmap, 0f, 0f, new Paint(Paint.FILTER_BITMAP_FLAG));
                document.finishPage(page);
                document.writeTo(output);
                runOnUiThread(() -> Toast.makeText(
                        this,
                        R.string.pdf_saved,
                        Toast.LENGTH_SHORT
                ).show());
            } catch (IOException exception) {
                runOnUiThread(() -> Toast.makeText(
                        this,
                        R.string.save_failed,
                        Toast.LENGTH_LONG
                ).show());
            } finally {
                document.close();
            }
        });
    }

    private void setBusy(boolean busy) {
        processingProgress.setVisibility(busy ? View.VISIBLE : View.GONE);
        documentPreview.setAlpha(busy ? 0.55f : 1f);
    }

    private void setOcrBusy(boolean busy) {
        ocrProgress.setVisibility(busy ? View.VISIBLE : View.GONE);
        recognizeTextButton.setEnabled(!busy && enhancedBitmap != null);
        recognizeTextButton.setText(busy ? R.string.recognizing : R.string.recognize_text);
    }

    private void updateControlState(boolean enabled) {
        originalButton.setEnabled(enabled);
        enhancedButton.setEnabled(enabled);
        saveImageButton.setEnabled(enabled);
        savePdfButton.setEnabled(enabled);
        recognizeTextButton.setEnabled(enabled);
        strengthSlider.setEnabled(enabled);
        findViewById(R.id.mode_group).setEnabled(enabled);
    }

    private void setCheckedMode(HandwritingEnhancer.Mode mode) {
        RadioGroup group = findViewById(R.id.mode_group);
        if (mode == HandwritingEnhancer.Mode.GRAYSCALE) {
            group.check(R.id.mode_grayscale);
        } else if (mode == HandwritingEnhancer.Mode.BLACK_AND_WHITE) {
            group.check(R.id.mode_black_white);
        } else {
            group.check(R.id.mode_natural);
        }
    }

    private void recycleBitmap(Bitmap bitmap) {
        if (bitmap != null && !bitmap.isRecycled()) {
            bitmap.recycle();
        }
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        if (selectedImageUri != null) {
            outState.putString(STATE_IMAGE_URI, selectedImageUri.toString());
        }
        outState.putString(STATE_MODE, selectedMode.name());
        outState.putInt(STATE_STRENGTH, strengthSlider.getProgress());
    }

    @Override
    protected void onDestroy() {
        processingGeneration++;
        cancelOcr();
        if (ocrEngine != null) {
            ocrEngine.close();
        }
        processingExecutor.shutdownNow();
        neuralEnhancer.close();
        documentPreview.setImageDrawable(null);
        super.onDestroy();
    }
}