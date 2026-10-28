/** 0.632 -> "63.2%" */
export function pct(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

/** 13.71 -> "13.7 ms", 1240 -> "1.24 s" */
export function ms(value: number): string {
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${value.toFixed(1)} ms`;
}

/** 0.70 -> "0.70" */
export function similarity(value: number): string {
  return value.toFixed(2);
}

/** Human labels for guard names reported by the API. */
export function guardLabel(guard: string): string {
  const labels: Record<string, string> = {
    negation: "Negation differs",
    numbers: "Numbers differ",
    opposite_action: "Opposite action",
  };
  return labels[guard] ?? guard.replace(/_/g, " ");
}
