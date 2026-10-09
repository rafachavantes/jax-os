"use client";

// The one toggle look for the whole app (spec: reuse AfkToggle's switch markup/styling
// everywhere a boolean setting needs a control). Presentational only — callers own the state.
type SwitchProps = {
  checked: boolean;
  onChange: () => void;
  disabled?: boolean;
  title?: string;
  "aria-label"?: string;
};

export function Switch({ checked, onChange, disabled, title, "aria-label": ariaLabel }: SwitchProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={ariaLabel}
      disabled={disabled}
      onClick={onChange}
      title={title}
      className={`relative h-6 w-11 flex-none rounded-full border transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
        checked ? "border-brand bg-brand" : "border-line-strong bg-surface-3"
      }`}
    >
      {/* Fix (AFK toggle overlap): explicit left-0.5 base — the knob's absolute position must not
          fall back to the browser's static-position resolution for left/right:auto. */}
      <span
        className={`absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-surface transition-transform ${
          checked ? "translate-x-[22px]" : "translate-x-0"
        }`}
      />
    </button>
  );
}
