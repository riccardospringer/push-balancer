import { promises as fs } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import YAML from 'yaml'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const frontendRoot = path.resolve(__dirname, '..')
const openapiPath = path.resolve(frontendRoot, '..', 'openapi.yaml')
const outputPath = path.resolve(
  frontendRoot,
  'src/api/generated/api-client-base.ts',
)

const document = YAML.parse(await fs.readFile(openapiPath, 'utf8'))

function resolveParameter(parameter) {
  if (!parameter?.$ref) {
    return parameter
  }

  const refName = parameter.$ref.split('/').at(-1)
  return document.components?.parameters?.[refName]
}

function pascalCase(value) {
  return value.charAt(0).toUpperCase() + value.slice(1)
}

const operations = []

for (const [routePath, pathItem] of Object.entries(document.paths ?? {})) {
  for (const method of ['get', 'post', 'put', 'patch', 'delete']) {
    const operation = pathItem?.[method]
    if (!operation?.operationId) {
      continue
    }

    const parameters = (operation.parameters ?? [])
      .map(resolveParameter)
      .filter(Boolean)
    const queryParams = parameters
      .filter((parameter) => parameter.in === 'query')
      .map((parameter) => parameter.name)

    operations.push({
      method: method.toUpperCase(),
      operationId: operation.operationId,
      path: routePath,
      queryParams,
      hasBody: ['POST', 'PUT', 'PATCH'].includes(method.toUpperCase()),
    })
  }
}

const lines = [
  '/*',
  ` * Auto-generated from ../openapi.yaml (OpenAPI ${document.openapi}, API ${document.info?.version ?? '0.0.0'}).`,
  ' * Run `pnpm generate:api-client` after changing the API spec.',
  ' */',
  '',
  "const BASE = import.meta.env.VITE_API_BASE ?? ''",
  '',
  'type QueryParams = Record<string, string | number | undefined>',
  '',
  'export interface ProblemDetails {',
  '  type?: string',
  '  title?: string',
  '  status?: number',
  '  detail?: string',
  '  instance?: string',
  '}',
  '',
  'export class ApiError extends Error {',
  '  status: number',
  '  problem?: ProblemDetails',
  '',
  '  constructor(status: number, message: string, problem?: ProblemDetails) {',
  '    super(message)',
  "    this.name = 'ApiError'",
  '    this.status = status',
  '    this.problem = problem',
  '  }',
  '}',
  '',
  'function withQuery(path: string, params: QueryParams = {}) {',
  '  const search = new URLSearchParams()',
  '  Object.entries(params).forEach(([key, value]) => {',
  "    if (value !== undefined && value !== '') {",
  '      search.set(key, String(value))',
  '    }',
  '  })',
  '  const query = search.toString()',
  '  return query ? `${path}?${query}` : path',
  '}',
  '',
  'async function request<T>(',
  '  path: string,',
  "  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE',",
  '  options: { body?: unknown; signal?: AbortSignal } = {},',
  '): Promise<T> {',
  '  const res = await fetch(`${BASE}${path}`, {',
  '    method,',
  '    signal: options.signal,',
  '    headers: {',
  "      Accept: 'application/json',",
  "      ...(options.body !== undefined ? { 'Content-Type': 'application/json' } : {}),",
  '    },',
  '    ...(options.body !== undefined ? { body: JSON.stringify(options.body) } : {}),',
  '  })',
  '',
  '  if (!res.ok) {',
  '    let problem: ProblemDetails | undefined',
  '    try {',
  '      problem = (await res.json()) as ProblemDetails',
  '    } catch {',
  '      problem = undefined',
  '    }',
  '    const message =',
  '      problem?.detail ??',
  '      problem?.title ??',
  '      `${res.status} ${res.statusText} - ${path}`',
  '    throw new ApiError(res.status, message, problem)',
  '  }',
  '',
  '  return res.json() as Promise<T>',
  '}',
  '',
  'export const rawClient = {',
]

for (const operation of operations) {
  const typeName = `${pascalCase(operation.operationId)}Params`
  if (operation.queryParams.length > 0) {
    lines.push(`  ${operation.operationId}: (`)
    lines.push(`    params: ${typeName} = {},`)
    lines.push('    signal?: AbortSignal,')
    lines.push('  ) =>')
    lines.push(
      `    request<unknown>(withQuery('${operation.path}', params), '${operation.method}', { signal }),`,
    )
    lines.push('')
    continue
  }

  if (operation.hasBody) {
    lines.push(
      `  ${operation.operationId}: (body: unknown = {}, signal?: AbortSignal) =>`,
    )
    lines.push(
      `    request<unknown>('${operation.path}', '${operation.method}', { body, signal }),`,
    )
    lines.push('')
    continue
  }

  lines.push(`  ${operation.operationId}: (signal?: AbortSignal) =>`)
  lines.push(
    `    request<unknown>('${operation.path}', '${operation.method}', { signal }),`,
  )
  lines.push('')
}

lines.push('} as const', '')

for (const operation of operations) {
  if (operation.queryParams.length === 0) {
    continue
  }

  const typeName = `${pascalCase(operation.operationId)}Params`
  lines.push(`export interface ${typeName} extends QueryParams {`)
  for (const parameterName of operation.queryParams) {
    lines.push(`  ${parameterName}?: string | number`)
  }
  lines.push('}', '')
}

await fs.mkdir(path.dirname(outputPath), { recursive: true })
await fs.writeFile(outputPath, `${lines.join('\n')}\n`, 'utf8')
