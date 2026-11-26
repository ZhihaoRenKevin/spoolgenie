# SpoolGenie Mobile App (iOS & Android)

SpoolGenie supports **two effortless ways** to run on iPhone, iPad, and Android phones while harnessing your Windows/macOS computer's local GPU (RTX 5090 / Apple Silicon) for heavy 2D SDXL, 3D Hunyuan3D-2, and parametric CAD generation:

---

## Option A — Zero-Compile Instant Mobile App (Recommended PWA)

1. On your Windows or macOS computer, launch **SpoolGenie Desktop** (`spoolgenie desktop`) or run:
   ```bash
   spoolgenie serve --lan --port 8765
   ```
2. Click **`📱 Mobile`** in the top bar to display the Wi-Fi QR code.
3. Scan with your phone camera:
   - **iOS (Safari)**: Tap **Share** → **Add to Home Screen** → Launches full-screen with bottom tab navigation (`💬 Chat` | `🎨 2D/3D` | `🔎 MakerWorld` | `⚙️ Specs & Export`).
   - **Android (Chrome)**: Tap **Menu (⋮)** → **Install app / Add to Home Screen**.

---

## Option B — Native iOS (`.ipa`) & Android (`.apk`) via Capacitor

Requires Node.js 20+, plus **Xcode** (for iOS) or **Android Studio** (for Android).

```bash
cd mobile
npm install

# Android APK
npm run android:add
npm run android:open   # Build APK or run on device in Android Studio

# iOS App (on macOS with Xcode)
npm run ios:add
npm run ios:open       # Archive or deploy to iPhone/iPad in Xcode
```

Once installed on your phone, enter your home PC/Mac LAN URL (e.g. `http://192.168.1.100:8765`) or any HTTPS tunnel URL once — SpoolGenie Mobile saves it automatically and reconnects straight to your studio.
