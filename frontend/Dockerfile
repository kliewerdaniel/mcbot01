# ==========================================
# Research Assistant GraphRAG - Frontend Dockerfile
# ==========================================

# Multi-stage build for optimized production image

# Build stage
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM node:18-alpine AS production

# Create app user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextapp -u 1001

# Set working directory
WORKDIR /app

# Copy built application from builder stage
COPY --from=builder --chown=nextapp:nodejs /app/.next ./.next
COPY --from=builder --chown=nextapp:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nextapp:nodejs /app/package.json ./package.json
COPY --from=builder --chown=nextapp:nodejs /app/public ./public
COPY --from=builder --chown=nextapp:nodejs /app/next.config.ts ./

# Switch to non-root user
USER nextapp

# Expose port
EXPOSE 3000

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3000
ENV NEXT_TELEMETRY_DISABLED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

# Start the application
CMD ["npm", "start"]
