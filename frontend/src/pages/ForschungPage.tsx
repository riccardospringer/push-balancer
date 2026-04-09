import { useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import {
  useForschung,
  useResearchRules,
  useMlStatus,
  useMlMonitoring,
  useGbrtStatus,
  useMlRetrain,
} from '@/hooks/useApi'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { StatCard } from '@/components/ui/StatCard'
import { Spinner } from '@/components/ui/Spinner'
import { Alert } from '@/components/ui/Alert'
import { Button } from '@/components/ui/Button'
import { Table, TableHeader, TableRow, TableCell } from '@/components/ui/Table'
import { fmtOR, fmtNum, fmtDateTime } from '@/lib/format'

export function ForschungPage() {
  const { data: forschung, isLoading: fLoading } = useForschung()
  const { data: rules } = useResearchRules()
  const { data: mlStatus } = useMlStatus()
  const { data: mlMonitoring } = useMlMonitoring()
  const { data: gbrtStatus } = useGbrtStatus()
  const retrainMutation = useMlRetrain()

  const [showAllRules, setShowAllRules] = useState(false)

  const visibleRules = rules?.rules
    ? showAllRules
      ? rules.rules
      : rules.rules.slice(0, 8)
    : []

  // Prepare chart data from recent predictions
  const chartData = mlMonitoring?.recentPredictions
    ?.filter((p) => p.actualOR != null)
    ?.slice(-30)
    ?.map((p, i) => ({
      i,
      predicted: +(p.predictedOR * 100).toFixed(2),
      actual: +((p.actualOR ?? 0) * 100).toFixed(2),
    })) ?? []

  return (
    <div
      style={{
        padding: '16px 24px',
        maxWidth: '1400px',
        margin: '0 auto',
        background: 'var(--bg)',
        animation: 'fadeIn 0.2s ease',
      }}
    >
      <div style={{ marginBottom: '20px' }}>
        <h1 style={{ fontSize: '18px', fontWeight: 700, margin: 0 }}>Forschung & ML</h1>
        <p style={{ fontSize: '12px', color: 'var(--text-secondary)', margin: '2px 0 0' }}>
          Modell-Performance, Research-Regeln und Learnings
        </p>
      </div>

      {fLoading && (
        <div style={{ padding: '60px', display: 'flex', justifyContent: 'center' }}>
          <Spinner size={28} />
        </div>
      )}

      {/* ML Status Row */}
      {mlStatus && (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
            gap: '12px',
            marginBottom: '20px',
          }}
        >
          <StatCard label="MAE Modell" value={fmtOR(mlStatus.mae)} accent />
          <StatCard label="RMSE" value={fmtOR(mlStatus.rmse)} />
          <StatCard label="R²" value={mlStatus.r2.toFixed(3)} />
          <StatCard
            label="Trainings-Rows"
            value={fmtNum(mlStatus.trainingRows)}
            sub={mlStatus.isEnsemble ? 'Stacking Ensemble' : 'Single LightGBM'}
          />
          <StatCard label="Modell-Version" value={mlStatus.modelVersion} sub={fmtDateTime(mlStatus.trainedAt)} />
        </div>
      )}

      {/* Retrain button */}
      {mlStatus && (
        <div style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Button
            variant="primary"
            onClick={() => retrainMutation.mutate()}
            disabled={retrainMutation.isPending}
          >
            {retrainMutation.isPending ? <Spinner size={14} color="#fff" /> : null}
            Modell neu trainieren
          </Button>
          {retrainMutation.isSuccess && (
            <Alert variant="success" style={{ flex: 1 }}>Training gestartet (Job ID: {(retrainMutation.data as any)?.jobId})</Alert>
          )}
          {retrainMutation.isError && (
            <Alert variant="error" style={{ flex: 1 }}>Training fehlgeschlagen.</Alert>
          )}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
        {/* Prediction vs Actual Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Predicted vs. Actual OR (letzte 30)</CardTitle>
            {mlMonitoring && (
              <Badge variant={mlMonitoring.drift > 0.01 ? 'amber' : 'green'}>
                MAE {fmtOR(mlMonitoring.rollingMAE)}
              </Badge>
            )}
          </CardHeader>
          <CardContent>
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="i" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
                  <YAxis
                    tick={{ fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    unit="%"
                    width={40}
                  />
                  <Tooltip
                    formatter={(v: number) => [`${v.toFixed(2)} %`]}
                    contentStyle={{ fontSize: '12px', borderRadius: '6px' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    stroke="var(--green)"
                    dot={false}
                    strokeWidth={2}
                    name="Ist-OR"
                  />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="var(--accent)"
                    dot={false}
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    name="Prognose"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div
                style={{
                  height: '200px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--text-tertiary)',
                  fontSize: '13px',
                }}
              >
                Noch keine Vergleichsdaten vorhanden
              </div>
            )}
          </CardContent>
        </Card>

        {/* A/B Test */}
        <Card>
          <CardHeader>
            <CardTitle>A/B-Test Status</CardTitle>
          </CardHeader>
          <CardContent>
            {forschung?.abTest ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '10px 14px',
                    background: 'var(--bg)',
                    borderRadius: 'var(--radius)',
                  }}
                >
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '13px' }}>Control</div>
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                      {forschung.abTest.control.name}
                    </div>
                  </div>
                  <Badge variant="default">MAE {fmtOR(forschung.abTest.control.mae)}</Badge>
                </div>
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '10px 14px',
                    background: 'var(--bg)',
                    borderRadius: 'var(--radius)',
                  }}
                >
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '13px' }}>Treatment</div>
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                      {forschung.abTest.treatment.name}
                    </div>
                  </div>
                  <Badge variant="default">MAE {fmtOR(forschung.abTest.treatment.mae)}</Badge>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                    p-Wert: {forschung.abTest.pValue.toFixed(3)}
                  </span>
                  {forschung.abTest.winner && (
                    <Badge variant="green">Gewinner: {forschung.abTest.winner}</Badge>
                  )}
                </div>
              </div>
            ) : (
              <div style={{ color: 'var(--text-tertiary)', fontSize: '13px' }}>
                Kein aktiver A/B-Test
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Research Rules */}
      {rules && (
        <Card style={{ marginBottom: '16px' }}>
          <CardHeader>
            <CardTitle>Was funktioniert (Research-Regeln)</CardTitle>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Badge variant="green">Rolling Accuracy {(rules.rollingAccuracy * 100).toFixed(1)} %</Badge>
              <Badge variant="default">v{rules.version}</Badge>
            </div>
          </CardHeader>
          <Table>
            <thead>
              <tr>
                <TableHeader>Kategorie</TableHeader>
                <TableHeader>Regel</TableHeader>
                <TableHeader>Konfidenz</TableHeader>
                <TableHeader>Support</TableHeader>
              </tr>
            </thead>
            <tbody>
              {visibleRules.map((rule) => (
                <TableRow key={rule.id}>
                  <TableCell>
                    <Badge variant="blue">{rule.category}</Badge>
                  </TableCell>
                  <TableCell style={{ maxWidth: '500px', color: 'var(--text-secondary)' }}>
                    {rule.rule}
                  </TableCell>
                  <TableCell>
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                      }}
                    >
                      <div
                        style={{
                          height: '4px',
                          width: '50px',
                          background: 'var(--border)',
                          borderRadius: '2px',
                          overflow: 'hidden',
                        }}
                      >
                        <div
                          style={{
                            height: '100%',
                            width: `${rule.confidence * 100}%`,
                            background: 'var(--accent)',
                            borderRadius: '2px',
                          }}
                        />
                      </div>
                      <span style={{ fontSize: '12px', fontVariantNumeric: 'tabular-nums' }}>
                        {(rule.confidence * 100).toFixed(0)} %
                      </span>
                    </div>
                  </TableCell>
                  <TableCell style={{ color: 'var(--text-secondary)' }}>{rule.supportCount}</TableCell>
                </TableRow>
              ))}
            </tbody>
          </Table>
          {rules.rules.length > 8 && (
            <div
              style={{
                padding: '10px 16px',
                borderTop: '1px solid var(--border-light)',
                display: 'flex',
                justifyContent: 'center',
              }}
            >
              <button
                onClick={() => setShowAllRules(!showAllRules)}
                style={{
                  fontFamily: 'inherit',
                  fontSize: '13px',
                  color: 'var(--accent)',
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                }}
              >
                {showAllRules ? 'Weniger anzeigen' : `Alle ${rules.rules.length} Regeln anzeigen`}
              </button>
            </div>
          )}
        </Card>
      )}

      {/* Learnings */}
      {forschung?.learnings && forschung.learnings.length > 0 && (
        <Card style={{ marginBottom: '16px' }}>
          <CardHeader>
            <CardTitle>Was wir gelernt haben</CardTitle>
          </CardHeader>
          <CardContent style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {forschung.learnings.map((l) => (
              <div
                key={l.id}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '10px',
                  padding: '10px 12px',
                  background: 'var(--bg)',
                  borderRadius: 'var(--radius)',
                  border: '1px solid var(--border-light)',
                }}
              >
                <Badge
                  variant={
                    l.impact === 'high' ? 'green' : l.impact === 'medium' ? 'amber' : 'default'
                  }
                >
                  {l.impact}
                </Badge>
                <span style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                  {l.text}
                </span>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* GBRT Status */}
      {gbrtStatus && (
        <Card>
          <CardHeader>
            <CardTitle>GBRT Modell (Pure Python)</CardTitle>
            <Badge variant={gbrtStatus.active ? 'green' : 'red'}>
              {gbrtStatus.active ? 'Aktiv' : 'Inaktiv'}
            </Badge>
          </CardHeader>
          <CardContent>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
                gap: '10px',
              }}
            >
              <div>
                <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '2px' }}>Version</div>
                <div style={{ fontWeight: 600 }}>{gbrtStatus.modelVersion}</div>
              </div>
              <div>
                <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '2px' }}>MAE</div>
                <div style={{ fontWeight: 600 }}>{fmtOR(gbrtStatus.mae)}</div>
              </div>
              <div>
                <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '2px' }}>
                  Trainings-Rows
                </div>
                <div style={{ fontWeight: 600 }}>{fmtNum(gbrtStatus.trainingRows)}</div>
              </div>
              <div>
                <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '2px' }}>
                  Letztes Training
                </div>
                <div style={{ fontWeight: 600 }}>{fmtDateTime(gbrtStatus.lastRetrain)}</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
