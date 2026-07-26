# InkClear

InkClear is an Android application for making handwritten document photos easier to read. It can scan a page with Google ML Kit's document scanner or import an existing image, enhance the image on the device, preview the original and enhanced versions, and export the result as JPEG or PDF.

## Features

- Import a handwritten image with the Android document picker.
- Scan one page with Google ML Kit Document Scanner when the device supports it.
- Reduce uneven lighting and shadows with a local background-normalization filter.
- Choose Natural, Grayscale, or Black & White output.
- Adjust enhancement strength from 0% to 100%.
- Save the enhanced result as `InkClear-enhanced.jpg` or `InkClear-enhanced.pdf`.
- Process images on-device; the app does not request storage permission.

## Requirements

- Android Studio with an Android SDK installation.
- JDK 11 or newer. The project compiles Java sources with Java 11 compatibility.
- Android SDK Platform 37.1 (the project uses compile SDK 37.1).
- A device or emulator running Android API 25 or newer.
- Internet access on the first Gradle build so Gradle and Maven dependencies can be downloaded.

The Gradle wrapper pins Gradle 9.6.1. The app uses Android Gradle Plugin 9.3.0, targets API 36, and has application ID `mobile_app.android_ai_integration`.

## Open and run in Android Studio

1. Open the repository root (`Android_AI_Integration`) in Android Studio.
2. Allow Gradle sync to finish and install any requested SDK components.
3. Select the `app` run configuration.
4. Start an API 25+ emulator or connect an Android device with USB debugging enabled.
5. Click **Run**. The launcher activity opens the InkClear home screen.

The ML Kit scanner may be unavailable on some emulators or devices. The **Choose image** path remains available as a fallback.

## Build from the command line

From the repository root on Windows PowerShell:

```powershell
.\gradlew.bat assembleDebug
```

The debug APK is written to `app/build/outputs/apk/debug/app-debug.apk`. Install it on a connected device with:

```powershell
.\gradlew.bat installDebug
```

On macOS/Linux, use `./gradlew` instead of `gradlew.bat`.

## Tests

Run local unit tests with:

```powershell
.\gradlew.bat test
```

Run the connected Android tests with an emulator or device attached:

```powershell
.\gradlew.bat connectedDebugAndroidTest
```

The repository contains a local unit test and an instrumented Android test. The instrumented package assertion has been corrected to `mobile_app.android_ai_integration`; both `test` and `connectedDebugAndroidTest` pass in the verified environment.

## Typical use

1. Tap **Choose a handwritten image**.
2. Select an image, or tap **AI scan** and scan one page.
3. Select **Natural**, **Grayscale**, or **B&W**.
4. Adjust **Strength** and compare **Original** with **Enhanced**.
5. Tap **Save image** or **Save PDF**, then choose a destination in the Android document picker.

## Project layout

```text
app/src/main/java/.../HomePage.java             Launcher screen
app/src/main/java/.../InkClearActivity.java    Import, scan, preview, and export flow
app/src/main/java/.../HandwritingEnhancer.java  On-device image enhancement
app/src/main/res/layout/                       XML layouts
app/src/main/res/values/                       Strings, colors, and theme
app/build.gradle.kts                            Android module configuration
gradle/libs.versions.toml                       Dependency and plugin versions
report/                                         LaTeX project report
```

## Report

The technical report source is in `report/report.tex`. From the repository root, compile it with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=report report/report.tex
```

This writes `report/report.pdf` and keeps auxiliary files in `report/`. The report documents the architecture, processing pipeline, build configuration, testing status, and limitations.
## Known issues and portability

The project can run on another computer when the required Android and Java tooling is installed. The Gradle wrapper downloads Gradle and project dependencies automatically, so a global Gradle installation is not required. The first build needs internet access.

Known issues:

- The instrumented example test previously used the template package ID `com.example.android_ai_integration`; its assertion has been corrected to the actual app ID `mobile_app.android_ai_integration`, and `connectedDebugAndroidTest` now passes on the tested emulator.
- `installDebug` installs the APK but does not open it. Start it from Android Studio or launch it from a terminal with `adb shell am start -n "mobile_app.android_ai_integration/.HomePage"`.
- The `libandroidx.graphics.path.so` strip message is a packaging warning. The APK was installed successfully and the warning does not prevent the app from running.
- ML Kit document scanning may be unavailable on some devices or emulators. Use **Choose image** when **AI scan** is unavailable.
- The app processes one page at a time and downsizes large images to a maximum dimension of 1800 pixels.

For a different Windows computer, select Android Studio's Embedded JDK (Java 17 or 21) under **Settings > Build, Execution, Deployment > Build Tools > Gradle**, install Android SDK Platform 37 and an API 25+ emulator, then run `gradlew.bat installDebug`. macOS and Linux use `./gradlew`. The repository's wrapper checksum must remain aligned with the official Gradle 9.6.1 binary.