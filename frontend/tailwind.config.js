/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: false, // Disable dark mode completely
  theme: {
    extend: {
      colors: {
        background: '#ffffff',
        foreground: '#171717',
      },
    },
  },
  plugins: [],
}; 