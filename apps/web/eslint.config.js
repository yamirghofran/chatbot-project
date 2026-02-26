// For more info, see https://github.com/storybookjs/eslint-plugin-storybook#configuration-flat-config-format
import storybook from "eslint-plugin-storybook";

import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([globalIgnores([
  'dist',
  'storybook-static',
  'src/routeTree.gen.ts',
]), {
  files: ['**/*.{ts,tsx}'],
  extends: [
    js.configs.recommended,
    tseslint.configs.recommended,
    reactHooks.configs.flat.recommended,
    reactRefresh.configs.vite,
  ],
  rules: {
    'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
  },
  languageOptions: {
    ecmaVersion: 2020,
    globals: globals.browser,
  },
}, {
  files: [
    '.storybook/preview.tsx',
    'src/components/ui/badge.tsx',
    'src/components/ui/button.tsx',
    'src/components/ui/tabs.tsx',
    'src/components/ui/toggle.tsx',
  ],
  rules: {
    'react-refresh/only-export-components': 'off',
  },
}, ...storybook.configs["flat/recommended"]])
