/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      colors: {
        'bg-primary': '#FAFAF8',
        'bg-secondary': '#F5F5F0',
        'accent-sage': '#9CAF88',
        'accent-sage-light': '#B8C4A8',
        'accent-sage-dark': '#7A8F6A',
        'accent-beige': '#E8E4D9',
        'accent-cream': '#F5F1E8',
        'text-primary': '#2C2C2A',
        'text-secondary': '#6B6B66',
        'text-muted': '#9A9A94',
        'card-bg': 'rgba(255, 255, 255, 0.7)',
        'code-bg': '#2D2D2D',
        'code-text': '#E8E4D9',
      },
      fontFamily: {
        serif: ['"Playfair Display"', 'Georgia', 'serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
      },
      borderRadius: {
        'sm': '12px',
        'md': '20px',
        'lg': '28px',
        'xl': '40px',
      },
      boxShadow: {
        'soft': '0 4px 20px rgba(0, 0, 0, 0.06)',
        'medium': '0 8px 32px rgba(0, 0, 0, 0.08)',
        'glow': '0 0 40px rgba(156, 175, 136, 0.15)',
      },
      animation: {
        'blink': 'blink 1s infinite',
      },
      keyframes: {
        blink: {
          '0%, 50%': { opacity: '1' },
          '51%, 100%': { opacity: '0' },
        },
      },
    },
  },
  plugins: [],
}
