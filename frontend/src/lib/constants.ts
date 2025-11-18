export const SITE_CONFIG = {
  name: 'Bot',
  url: process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:3000',
  description: 'AI-Powered Research and Analysis Platform',
  ogImage: '/og-image.png',
} as const;
