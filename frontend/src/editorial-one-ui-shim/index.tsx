/* eslint-disable react-refresh/only-export-components */
import type { PropsWithChildren } from 'react'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  FilterChip,
  Input,
  Modal,
  Select,
  Spinner,
  StatCard,
  StatusDot,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@/components/ui'

export const theme = {
  name: 'editorial-one-ui-shim',
}

export function ThemeProvider({
  children,
}: PropsWithChildren<{ theme?: unknown }>) {
  return <>{children}</>
}

export function CssBaseline() {
  return null
}

export function AppBar({ children }: PropsWithChildren) {
  return (
    <div
      style={{
        background: 'var(--white)',
        borderBottom: '1px solid var(--border)',
        position: 'sticky',
        top: 0,
        zIndex: 100,
      }}
    >
      {children}
    </div>
  )
}

export function Toolbar({ children }: PropsWithChildren) {
  return (
    <div
      style={{
        padding: '0 24px',
        minHeight: '56px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}
    >
      {children}
    </div>
  )
}

export {
  Alert,
  Badge,
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  FilterChip,
  Input,
  Modal,
  Select,
  Spinner,
  StatCard,
  StatusDot,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
}

export type AlertSeverity = 'info' | 'success' | 'warning' | 'error'
