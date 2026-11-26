import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'io.spoolgenie.studio',
  appName: 'SpoolGenie',
  webDir: 'www',
  server: {
    // Allow connecting to your home PC/Mac SpoolGenie server over local Wi-Fi HTTP
    cleartext: true,
    androidScheme: 'http',
  },
  ios: {
    contentInset: 'automatic',
    allowsLinkPreview: false,
  },
  android: {
    allowMixedContent: true,
  },
};

export default config;
