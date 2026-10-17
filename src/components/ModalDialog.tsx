"use client";

import { useEffect, useRef, type ComponentProps } from "react";

// Pending-close guard shared by every modal: while a write is pending Escape stays inert, like
// the disabled buttons. Exported pure so a no-jsdom test can assert it.
export function requestClose(pending: boolean, close: () => void) {
  if (!pending) close();
}

// Native modal <dialog>: opens on mount; Escape never closes it natively, it goes through onClose
// unless `pending`. Everything else (aria-*, className, onClick, children) passes straight through.
export function ModalDialog({
  pending = false,
  onClose,
  ...rest
}: Omit<ComponentProps<"dialog">, "onClose" | "onCancel" | "ref"> & { pending?: boolean; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dlg = ref.current;
    if (dlg && typeof dlg.showModal === "function") dlg.showModal();
  }, []);
  return (
    <dialog
      {...rest}
      ref={ref}
      onClose={onClose}
      onCancel={(e) => {
        e.preventDefault();
        requestClose(pending, onClose);
      }}
    />
  );
}
