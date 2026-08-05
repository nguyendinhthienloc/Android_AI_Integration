package mobile_app.android_ai_integration;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/** Pure text normalization used by OCR UI and tests. */
final class OcrTextFormatter {

    private static final Pattern HORIZONTAL_WHITESPACE = Pattern.compile("[\\t\\x0B\\f ]+");

    private OcrTextFormatter() {
    }

    static String formatLines(List<String> lines) {
        if (lines == null || lines.isEmpty()) {
            return "";
        }
        return normalize(String.join("\n", lines));
    }

    static String normalize(String text) {
        if (text == null || text.isEmpty()) {
            return "";
        }

        String normalizedNewlines = text.replace("\r\n", "\n").replace('\r', '\n');
        String[] sourceLines = normalizedNewlines.split("\n", -1);
        List<String> result = new ArrayList<>();
        boolean previousBlank = true;

        for (String sourceLine : sourceLines) {
            String line = HORIZONTAL_WHITESPACE.matcher(sourceLine).replaceAll(" ").trim();
            boolean blank = line.isEmpty();
            if (blank) {
                if (!previousBlank) {
                    result.add("");
                }
            } else {
                result.add(line);
            }
            previousBlank = blank;
        }

        while (!result.isEmpty() && result.get(result.size() - 1).isEmpty()) {
            result.remove(result.size() - 1);
        }
        return String.join("\n", result);
    }
}