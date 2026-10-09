const { heroui } = require("@heroui/react");

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./node_modules/@heroui/theme/dist/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        surface: 'var(--sa-surface)',
        'surface-soft': 'var(--sa-surface-soft)',
        'surface-strong': 'var(--sa-surface-strong)',
        text: 'var(--sa-text)',
        'text-soft': 'var(--sa-text-soft)',
        'text-faint': 'var(--sa-text-faint)',
        border: 'var(--sa-border)',
        'border-strong': 'var(--sa-border-strong)',
        accent: 'var(--sa-accent)',
        'accent-strong': 'var(--sa-accent-strong)',
        'accent-soft': 'var(--sa-accent-soft)',
        warning: 'var(--sa-warning)',
        danger: 'var(--sa-danger)',
        success: 'var(--sa-success)',
      },
      borderRadius: {
        sm: 'var(--sa-radius-sm)',
        md: 'var(--sa-radius-md)',
        lg: 'var(--sa-radius-lg)',
      },
      spacing: {
        'sa-1': 'var(--sa-space-1)',
        'sa-2': 'var(--sa-space-2)',
        'sa-3': 'var(--sa-space-3)',
        'sa-4': 'var(--sa-space-4)',
        'sa-5': 'var(--sa-space-5)',
        'sa-6': 'var(--sa-space-6)',
        'sa-7': 'var(--sa-space-7)',
      },
      fontFamily: {
        sans: 'var(--sa-font-sans)',
        display: 'var(--sa-font-display)',
        mono: 'var(--sa-font-mono)',
      },
      boxShadow: {
        sm: 'var(--sa-shadow-sm)',
        md: 'var(--sa-shadow-md)',
        lg: 'var(--sa-shadow-lg)',
      },
    },
  },
  darkMode: "class",
  plugins: [heroui()],
};
