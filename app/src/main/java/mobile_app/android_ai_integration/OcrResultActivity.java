package mobile_app.android_ai_integration;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.net.Uri;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.ComponentActivity;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class OcrResultActivity extends ComponentActivity {

    public static final String EXTRA_TEXT = "ocr_text";
    public static final String EXTRA_LINE_COUNT = "ocr_line_count";
    public static final String EXTRA_DURATION_MS = "ocr_duration_ms";
    public static final String EXTRA_AVERAGE_CONFIDENCE = "ocr_average_confidence";

    private static final String STATE_EDITED_TEXT = "edited_text";

    private final ExecutorService fileExecutor = Executors.newSingleThreadExecutor();

    private EditText recognizedText;
    private TextView characterCount;
    private String pendingText = "";

    private final ActivityResultLauncher<String> textSaver = registerForActivityResult(
            new ActivityResultContracts.CreateDocument("text/plain"),
            uri -> {
                if (uri != null) {
                    saveText(uri, pendingText);
                }
            }
    );

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.ocr_result_activity);

        recognizedText = findViewById(R.id.recognized_text);
        characterCount = findViewById(R.id.character_count);

        String initialText = savedInstanceState == null
                ? getIntent().getStringExtra(EXTRA_TEXT)
                : savedInstanceState.getString(STATE_EDITED_TEXT, "");
        recognizedText.setText(initialText == null ? "" : initialText);
        recognizedText.setSelection(recognizedText.length());
        updateCharacterCount();

        configurePreview();
        configureMetadata();
        configureActions();
        configureTextUpdates();

        if (recognizedText.length() == 0) {
            recognizedText.setHint(R.string.no_text_detected_hint);
        }
    }

    private void configurePreview() {
        ImageView preview = findViewById(R.id.ocr_source_preview);
        Uri source = getIntent().getData();
        if (source == null) {
            preview.setVisibility(View.GONE);
            return;
        }

        try {
            preview.setImageURI(source);
        } catch (RuntimeException exception) {
            preview.setVisibility(View.GONE);
        }
    }

    private void configureMetadata() {
        int lines = getIntent().getIntExtra(EXTRA_LINE_COUNT, 0);
        long durationMs = getIntent().getLongExtra(EXTRA_DURATION_MS, 0L);
        float confidence = getIntent().getFloatExtra(EXTRA_AVERAGE_CONFIDENCE, 0f);
        TextView metadata = findViewById(R.id.ocr_metadata);
        metadata.setText(getString(
                R.string.ocr_metadata,
                lines,
                durationMs / 1000f,
                Math.round(confidence * 100f)
        ));
    }

    private void configureActions() {
        findViewById(R.id.ocr_back_button).setOnClickListener(view -> finish());
        findViewById(R.id.copy_text_button).setOnClickListener(view -> copyText());
        findViewById(R.id.save_text_button).setOnClickListener(view -> {
            pendingText = recognizedText.getText().toString();
            textSaver.launch("InkClear-recognized.txt");
        });
        findViewById(R.id.recognize_again_button).setOnClickListener(view -> {
            setResult(Activity.RESULT_OK);
            finish();
        });
    }

    private void configureTextUpdates() {
        recognizedText.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence text, int start, int count, int after) {
            }

            @Override
            public void onTextChanged(CharSequence text, int start, int before, int count) {
                updateCharacterCount();
            }

            @Override
            public void afterTextChanged(Editable editable) {
            }
        });
    }

    private void updateCharacterCount() {
        characterCount.setText(getString(
                R.string.character_count,
                recognizedText.length()
        ));
    }

    private void copyText() {
        String text = recognizedText.getText().toString();
        ClipboardManager clipboard = getSystemService(ClipboardManager.class);
        clipboard.setPrimaryClip(ClipData.newPlainText(getString(R.string.recognized_text), text));
        Toast.makeText(this, R.string.text_copied, Toast.LENGTH_SHORT).show();
    }

    private void saveText(Uri destination, String text) {
        fileExecutor.execute(() -> {
            try (OutputStream output = getContentResolver().openOutputStream(destination)) {
                if (output == null) {
                    throw new IOException("Unable to open destination");
                }
                output.write(text.getBytes(StandardCharsets.UTF_8));
                output.flush();
                runOnUiThread(() -> Toast.makeText(
                        this,
                        R.string.text_saved,
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

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        outState.putString(STATE_EDITED_TEXT, recognizedText.getText().toString());
    }

    @Override
    protected void onDestroy() {
        fileExecutor.shutdownNow();
        super.onDestroy();
    }
}