import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      colors: {
        brand: {
          50:  "#eef2ff",
          100: "#e0e7ff",
          400: "#818cf8",
          500: "#6366f1",
          600: "#4f46e5",
          700: "#4338ca",
        },
      },
      backgroundImage: {
        "grad-card":   "linear-gradient(135deg, #1e293b 0%, #0f172a 100%)",
        "grad-indigo": "linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)",
        "grad-purple": "linear-gradient(135deg, #7c3aed 0%, #ec4899 100%)",
        "grad-cyan":   "linear-gradient(135deg, #0891b2 0%, #6366f1 100%)",
        "grad-green":  "linear-gradient(135deg, #059669 0%, #0891b2 100%)",
        "grad-amber":  "linear-gradient(135deg, #d97706 0%, #dc2626 100%)",
      },
      boxShadow: {
        card:   "0 1px 3px 0 rgba(0,0,0,.4), 0 1px 2px -1px rgba(0,0,0,.4)",
        glow:   "0 0 20px rgba(99,102,241,.35)",
        "glow-sm": "0 0 10px rgba(99,102,241,.2)",
      },
      animation: {
        "fade-in": "fadeIn .25s ease-out",
        "slide-up": "slideUp .3s ease-out",
      },
      keyframes: {
        fadeIn:  { from: { opacity: "0" }, to: { opacity: "1" } },
        slideUp: { from: { opacity: "0", transform: "translateY(10px)" }, to: { opacity: "1", transform: "translateY(0)" } },
      },
    },
  },
  plugins: [],
};

export default config;
