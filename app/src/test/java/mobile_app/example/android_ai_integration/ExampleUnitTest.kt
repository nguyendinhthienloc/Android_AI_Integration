package mobile_app.android_ai_integration

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class OcrTextFormatterTest {

    @Test
    fun normalize_collapsesHorizontalWhitespaceAndPreservesParagraphs() {
        val input = "  First\t line  \r\n\r\n\r\n Second   line "
        assertEquals("First line\n\nSecond line", OcrTextFormatter.normalize(input))
    }

    @Test
    fun formatLines_handlesEmptyAndUnicodeText() {
        assertEquals("", OcrTextFormatter.formatLines(emptyList()))
        assertEquals(
            "Café déjà vu\nTiếng Việt",
            OcrTextFormatter.formatLines(listOf(" Café  déjà vu ", "Tiếng Việt"))
        )
    }

    @Test
    fun normalize_trimsLeadingAndTrailingBlankLines() {
        assertEquals("InkClear", OcrTextFormatter.normalize("\n\n InkClear \n\n"))
    }

    @Test
    fun sigmoid_isStableAtUsefulExtremes() {
        assertEquals(0.5f, NeuralDocumentEnhancer.sigmoid(0f), 0.00001f)
        assertTrue(NeuralDocumentEnhancer.sigmoid(12f) > 0.999f)
        assertTrue(NeuralDocumentEnhancer.sigmoid(-12f) < 0.001f)
    }
}