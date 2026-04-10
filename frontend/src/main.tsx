import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import {
  CssBaseline,
  ThemeProvider,
  theme,
} from '@spring-media/editorial-one-ui'
import '@spring-media/editorial-one-ui/fonts.css'
import App from './app'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </StrictMode>,
)
