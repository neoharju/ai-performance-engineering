'use client';

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { getThemes } from './api';

export type ColorMode = 'dark' | 'light' | 'system';

export interface ThemeColors {
  primary: string;
  secondary: string;
  bg: string;
  card: string;
  success?: string;
  warning?: string;
  danger?: string;
  info?: string;
  text?: string;
  textSecondary?: string;
  textMuted?: string;
  border?: string;
}

export interface Theme {
  id: string;
  name: string;
  description?: string;
  colors: ThemeColors;
}

// Built-in themes
export const builtInThemes: Theme[] = [
  {
    id: 'cyberpunk',
    name: 'Cyberpunk',
    description: 'Neon cyan and purple on dark background',
    colors: {
      primary: '#00f5d4',
      secondary: '#9d4edd',
      bg: '#06060a',
      card: 'rgba(16, 16, 24, 0.9)',
    },
  },
  {
    id: 'midnight-blue',
    name: 'Midnight Blue',
    description: 'Deep blue with electric accents',
    colors: {
      primary: '#4cc9f0',
      secondary: '#7209b7',
      bg: '#0a1628',
      card: 'rgba(15, 30, 50, 0.9)',
    },
  },
  {
    id: 'forest',
    name: 'Forest',
    description: 'Natural greens with warm highlights',
    colors: {
      primary: '#00f5a0',
      secondary: '#ffc43d',
      bg: '#0a140a',
      card: 'rgba(10, 25, 15, 0.9)',
    },
  },
  {
    id: 'sunset',
    name: 'Sunset',
    description: 'Warm oranges and pinks',
    colors: {
      primary: '#ff6b6b',
      secondary: '#ffc43d',
      bg: '#1a0a0a',
      card: 'rgba(30, 15, 15, 0.9)',
    },
  },
  {
    id: 'monochrome',
    name: 'Monochrome',
    description: 'Clean grayscale aesthetic',
    colors: {
      primary: '#ffffff',
      secondary: '#888888',
      bg: '#0a0a0a',
      card: 'rgba(20, 20, 20, 0.9)',
    },
  },
  {
    id: 'aurora',
    name: 'Aurora',
    description: 'Northern lights inspired',
    colors: {
      primary: '#7ee8fa',
      secondary: '#80ff72',
      bg: '#050510',
      card: 'rgba(10, 15, 25, 0.9)',
    },
  },
  {
    id: 'light',
    name: 'Light Mode',
    description: 'Bright background with crisp text',
    colors: {
      primary: '#7c3aed',
      secondary: '#22c55e',
      bg: '#f8fafc',
      card: '#ffffff',
      text: '#0f172a',
      textSecondary: '#334155',
      textMuted: '#475569',
      success: '#16a34a',
      warning: '#d97706',
      danger: '#dc2626',
      info: '#2563eb',
      border: 'rgba(15, 23, 42, 0.12)',
    },
  },
];

interface ThemeContextValue {
  currentTheme: Theme;
  colorMode: ColorMode;
  resolvedColorMode: Exclude<ColorMode, 'system'>;
  themes: Theme[];
  setTheme: (themeId: string) => void;
  setColorMode: (mode: ColorMode) => void;
  availableThemes: Theme[];
  selectedThemeId: string;
  setSelectedThemeId: (themeId: string) => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

const defaultColors = {
  primary: '#00f5d4',
  secondary: '#9d4edd',
  bg: '#06060a',
  card: 'rgba(16, 16, 24, 0.9)',
  success: '#00f5a0',
  warning: '#ffc43d',
  danger: '#ff4757',
  info: '#4cc9f0',
  text: '#ffffff',
  textSecondary: 'rgba(255,255,255,0.8)',
  textMuted: 'rgba(255,255,255,0.6)',
  border: 'rgba(255, 255, 255, 0.06)',
};

function parseRgb(color?: string): [number, number, number] | null {
  if (!color) return null;
  const normalized = color.trim();
  if (normalized.startsWith('#')) {
    let hex = normalized.slice(1);
    if (hex.length === 3) {
      hex = hex
        .split('')
        .map((c) => c + c)
        .join('');
    }
    if (hex.length !== 6) return null;
    const num = parseInt(hex, 16);
    return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
  }
  const rgbMatch = normalized.match(/rgba?\((\d+)[,\s]+(\d+)[,\s]+(\d+)/i);
  if (rgbMatch) {
    return [parseInt(rgbMatch[1], 10), parseInt(rgbMatch[2], 10), parseInt(rgbMatch[3], 10)];
  }
  const plainMatch = normalized.match(/^(\d{1,3})[,\s]+(\d{1,3})[,\s]+(\d{1,3})$/);
  if (plainMatch) {
    return [parseInt(plainMatch[1], 10), parseInt(plainMatch[2], 10), parseInt(plainMatch[3], 10)];
  }
  return null;
}

function toRgbString(color?: string, fallback?: string) {
  const rgb = parseRgb(color) ?? parseRgb(fallback || '');
  return rgb ? `${rgb[0]}, ${rgb[1]}, ${rgb[2]}` : '';
}

function toRgba(color?: string, alpha = 1, fallback?: string) {
  const rgb = parseRgb(color) ?? parseRgb(fallback || '');
  return rgb ? `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})` : fallback || '';
}

function normalizeApiTheme(theme: any): Theme {
  const colors = theme?.colors || {};
  return {
    id: theme?.id || theme?.name || 'custom',
    name: theme?.name || theme?.id || 'Custom Theme',
    description: theme?.description,
    colors: {
      primary: colors.accent_primary || colors.primary || colors.accent || defaultColors.primary,
      secondary: colors.accent_secondary || colors.secondary || defaultColors.secondary,
      bg: colors.bg_primary || colors.bg || defaultColors.bg,
      card: colors.bg_card || colors.card || defaultColors.card,
      success: colors.accent_success,
      warning: colors.accent_warning,
      danger: colors.accent_danger,
      info: colors.accent_info,
      text: colors.text_primary,
      textSecondary: colors.text_secondary,
      textMuted: colors.text_muted,
      border: colors.border,
    },
  };
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [currentThemeId, setCurrentThemeId] = useState('cyberpunk');
  const [colorMode, setColorMode] = useState<ColorMode>('dark');
  const [resolvedColorMode, setResolvedColorMode] = useState<Exclude<ColorMode, 'system'>>('dark');
  const [themes, setThemes] = useState<Theme[]>(builtInThemes);

  // Load saved theme on mount
  useEffect(() => {
    try {
      const savedThemeId = localStorage.getItem('dashboard_theme');
      const savedColorMode = localStorage.getItem('dashboard_color_mode');
      if (savedThemeId) setCurrentThemeId(savedThemeId);
      if (savedColorMode) setColorMode(savedColorMode as 'dark' | 'light' | 'system');
    } catch {
      // Ignore localStorage errors
    }
  }, []);

  // Load available themes from backend
  useEffect(() => {
    let mounted = true;
    const loadThemes = async () => {
      try {
        const response = await getThemes();
        if (!mounted || !response?.themes) return;
        const apiThemes = (response.themes as any[]).map(normalizeApiTheme);
        const merged: Theme[] = [];
        [...builtInThemes, ...apiThemes].forEach((theme) => {
          if (!merged.find((t) => t.id === theme.id)) {
            merged.push(theme);
          }
        });
        setThemes(merged);
        const savedThemeId = (() => {
          try {
            return localStorage.getItem('dashboard_theme');
          } catch {
            return null;
          }
        })();
        const desiredThemeId =
          (savedThemeId && merged.find((t) => t.id === savedThemeId)?.id) ||
          (merged.find((t) => t.id === currentThemeId)?.id ? currentThemeId : undefined) ||
          response.current ||
          merged[0]?.id ||
          'cyberpunk';
        if (desiredThemeId && desiredThemeId !== currentThemeId) {
          setCurrentThemeId(desiredThemeId);
        }
      } catch (err) {
        console.warn('Failed to load themes', err);
      }
    };
    loadThemes();
    return () => {
      mounted = false;
    };
  }, []);

  // Track system preference when using "system" mode
  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const resolve = () => (colorMode === 'system' ? (media.matches ? 'dark' : 'light') : colorMode);
    setResolvedColorMode(resolve());

    const handler = (event: MediaQueryListEvent) => {
      if (colorMode === 'system') {
        setResolvedColorMode(event.matches ? 'dark' : 'light');
      }
    };
    media.addEventListener('change', handler);
    return () => media.removeEventListener('change', handler);
  }, [colorMode]);

  // Keep a light-friendly theme selected when switching modes
  useEffect(() => {
    if (!themes.length) return;
    const hasLightTheme = themes.find((t) => t.id.toLowerCase().includes('light'));
    const hasDarkTheme = themes.find((t) => !t.id.toLowerCase().includes('light'));

    if (resolvedColorMode === 'light' && hasLightTheme && !currentThemeId.toLowerCase().includes('light')) {
      setCurrentThemeId(hasLightTheme.id);
    }
    if (resolvedColorMode === 'dark' && hasDarkTheme && currentThemeId.toLowerCase().includes('light')) {
      setCurrentThemeId(hasDarkTheme.id);
    }
  }, [resolvedColorMode, themes, currentThemeId]);

  // Apply theme CSS variables whenever theme changes
  useEffect(() => {
    const theme = themes.find((t) => t.id === currentThemeId) || themes[0];
    const root = document.documentElement;
    const colors = {
      ...defaultColors,
      ...(theme?.colors || {}),
    };
    const effectiveMode = resolvedColorMode || 'dark';

    root.dataset.colorMode = effectiveMode;

    root.style.setProperty('--accent-primary', colors.primary);
    root.style.setProperty('--accent-secondary', colors.secondary);
    root.style.setProperty('--accent-success', colors.success || defaultColors.success);
    root.style.setProperty('--accent-warning', colors.warning || defaultColors.warning);
    root.style.setProperty('--accent-danger', colors.danger || defaultColors.danger);
    root.style.setProperty('--accent-info', colors.info || defaultColors.info);

    root.style.setProperty('--accent-primary-rgb', toRgbString(colors.primary, defaultColors.primary));
    root.style.setProperty('--accent-secondary-rgb', toRgbString(colors.secondary, defaultColors.secondary));
    root.style.setProperty('--accent-success-rgb', toRgbString(colors.success, defaultColors.success));
    root.style.setProperty('--accent-warning-rgb', toRgbString(colors.warning, defaultColors.warning));
    root.style.setProperty('--accent-danger-rgb', toRgbString(colors.danger, defaultColors.danger));
    root.style.setProperty('--accent-info-rgb', toRgbString(colors.info, defaultColors.info));

    const bgColor = colors.bg || defaultColors.bg;
    root.style.setProperty('--bg-primary', bgColor);
    root.style.setProperty('--bg-primary-90', toRgba(bgColor, 0.9, defaultColors.bg));
    root.style.setProperty('--bg-primary-50', toRgba(bgColor, 0.5, defaultColors.bg));

    const cardColor = colors.card || defaultColors.card;
    root.style.setProperty('--bg-card', cardColor);
    root.style.setProperty('--bg-card-90', toRgba(cardColor, 0.9, defaultColors.card));
    root.style.setProperty('--bg-card-50', toRgba(cardColor, 0.5, defaultColors.card));

    const textPrimary = colors.text || (effectiveMode === 'light' ? '#0f172a' : defaultColors.text);
    const textSecondary = colors.textSecondary || (effectiveMode === 'light' ? '#334155' : defaultColors.textSecondary);
    const textMuted = colors.textMuted || (effectiveMode === 'light' ? '#475569' : defaultColors.textMuted);

    root.style.setProperty('--text-primary', textPrimary);
    root.style.setProperty('--text-secondary', textSecondary);
    root.style.setProperty('--text-muted', textMuted);

    root.style.setProperty('--text-primary-rgb', toRgbString(textPrimary, defaultColors.text));
    root.style.setProperty('--text-secondary-rgb', toRgbString(textSecondary, defaultColors.textSecondary));
    root.style.setProperty('--text-muted-rgb', toRgbString(textMuted, defaultColors.textMuted));

    const borderColor =
      colors.border ||
      (effectiveMode === 'light' ? 'rgba(15, 23, 42, 0.12)' : defaultColors.border);
    root.style.setProperty('--border-glass', borderColor);

    // Also update body background
    document.body.style.backgroundColor = bgColor;

    // Save to localStorage
    try {
      localStorage.setItem('dashboard_theme', currentThemeId);
      localStorage.setItem('dashboard_color_mode', colorMode);
    } catch {
      // Ignore localStorage errors
    }
  }, [currentThemeId, colorMode, themes, resolvedColorMode]);

  const setTheme = useCallback((themeId: string) => {
    setCurrentThemeId(themeId);
  }, []);

  const currentTheme = themes.find((t) => t.id === currentThemeId) || themes[0];

  return (
    <ThemeContext.Provider
      value={{
        currentTheme,
        colorMode,
        themes,
        setTheme,
        setColorMode,
        availableThemes: themes,
        selectedThemeId: currentThemeId,
        setSelectedThemeId: setTheme,
        resolvedColorMode,
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}



