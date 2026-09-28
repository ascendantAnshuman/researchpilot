/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Produces a self-contained server in .next/standalone for the Docker image (Phase 8).
  output: "standalone",
  poweredByHeader: false,
};

export default nextConfig;
