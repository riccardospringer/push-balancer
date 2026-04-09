import js from '@eslint/js'
import tsParser from '@typescript-eslint/parser'
import tsPlugin from '@typescript-eslint/eslint-plugin'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'

const browserGlobals = {
  AbortController: 'readonly',
  Event: 'readonly',
  HTMLElement: 'readonly',
  KeyboardEvent: 'readonly',
  URL: 'readonly',
  URLSearchParams: 'readonly',
  clearTimeout: 'readonly',
  console: 'readonly',
  document: 'readonly',
  fetch: 'readonly',
  navigator: 'readonly',
  setTimeout: 'readonly',
  window: 'readonly',
}

export default [
  js.configs.recommended,
  {
    ignores: ['dist/**', 'node_modules/**'],
  },
  {
    files: ['src/**/*.{ts,tsx}', 'vite.config.ts'],
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        ecmaFeatures: {
          jsx: true,
        },
      },
      ecmaVersion: 'latest',
      sourceType: 'module',
      globals: browserGlobals,
    },
    plugins: {
      '@typescript-eslint': tsPlugin,
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...tsPlugin.configs.recommended.rules,
      ...reactHooks.configs.recommended.rules,
      'no-undef': 'off',
      '@typescript-eslint/consistent-type-imports': 'error',
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
        },
      ],
      'react-hooks/set-state-in-effect': 'off',
      'react-refresh/only-export-components': 'warn',
    },
  },
]
