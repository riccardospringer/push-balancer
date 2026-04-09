/**
 * Editorial One UI compatibility layer.
 *
 * The project is configured for GitHub Package Registry via `frontend/.npmrc`.
 * Once `NPM_TOKEN` is available, replace this file with:
 *
 *   export * from '@spring-media/editorial-one-ui'
 *
 * Until then, local fallback components preserve the app contract and keep the
 * migration surface limited to this module.
 */

export { Button } from './Button'
export { Badge } from './Badge'
export { Card, CardHeader, CardTitle, CardContent } from './Card'
export { Spinner } from './Spinner'
export { StatusDot } from './StatusDot'
export { StatCard } from './StatCard'
export { FilterChip } from './FilterChip'
export { Table, TableHeader, TableRow, TableCell } from './Table'
export { Tabs, TabsList, TabsTrigger, TabsContent } from './Tabs'
export { Input } from './Input'
export { Select } from './Select'
export { Modal } from './Modal'
export { Alert } from './Alert'
