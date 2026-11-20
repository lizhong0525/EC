# syntax=docker/dockerfile:1.7

FROM node:22-alpine AS builder

WORKDIR /build/web

COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./
RUN npm run build


FROM nginxinc/nginx-unprivileged:1.27-alpine AS runtime

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /build/web/dist /usr/share/nginx/html

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=6 \
    CMD ["wget", "--quiet", "--output-document=-", "http://127.0.0.1:8080/healthz"]

# Avoid entrypoint scripts that try to rewrite configuration on a read-only
# root filesystem. The reviewed configuration above is the only active config.
ENTRYPOINT []
CMD ["nginx", "-g", "daemon off;"]
