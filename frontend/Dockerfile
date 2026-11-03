# Frontend image — Enterprise AI Intelligence Platform (Next.js)
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package*.json* ./
RUN npm install --legacy-peer-deps

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# Build-time env: the backend URL the frontend proxies to. In production you
# typically call the backend directly via NEXT_PUBLIC_API_URL (set in lib/api.ts).
ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
CMD ["node", "server.js"]
