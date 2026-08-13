import { useEffect, useRef, useState } from "react";
import type { ButtonHTMLAttributes, KeyboardEvent as ReactKeyboardEvent, ReactNode } from "react";
import { AlertCircle, Check, ChevronDown, LoaderCircle, RotateCw } from "lucide-react";

export function Button({
  className = "",
  variant = "primary",
  busy = false,
  children,
  disabled,
  type = "button",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  busy?: boolean;
}) {
  return (
    <button
      type={type}
      className={`button button-${variant} ${className}`}
      disabled={disabled || busy}
      {...props}
    >
      {busy && <LoaderCircle className="spin" size={16} aria-hidden="true" />}
      {children}
    </button>
  );
}

export function Switch({
  checked,
  onChange,
  label,
  description,
  disabled = false,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  description?: string;
  disabled?: boolean;
}) {
  return (
    <label className={`switch-row ${disabled ? "is-disabled" : ""}`}>
      <span>
        <strong>{label}</strong>
        {description && <small>{description}</small>}
      </span>
      <input
        className="sr-only"
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span className="switch" aria-hidden="true">
        <span />
      </span>
    </label>
  );
}

export function Segmented<T extends string>({
  value,
  onChange,
  label,
  options,
}: {
  value: T;
  onChange: (value: T) => void;
  label: string;
  options: Array<{ value: T; label: string; description?: string }>;
}) {
  return (
    <fieldset className="segmented-field">
      <legend className="sr-only">{label}</legend>
      <div className="segmented" role="radiogroup" aria-label={label}>
        {options.map((option) => (
          <label
            className={`segment ${value === option.value ? "is-selected" : ""}`}
            key={option.value}
          >
            <input
              className="sr-only"
              type="radio"
              name={label}
              value={option.value}
              checked={value === option.value}
              onChange={() => onChange(option.value)}
            />
            <span>{option.label}</span>
            {option.description && <small>{option.description}</small>}
            {value === option.value && <Check size={15} aria-hidden="true" />}
          </label>
        ))}
      </div>
    </fieldset>
  );
}

export function SelectMenu<T extends string>({
  id,
  value,
  onChange,
  options,
  disabled = false,
  ariaLabel,
}: {
  id: string;
  value: T;
  onChange: (value: T) => void;
  options: Array<{ value: T; label: string }>;
  disabled?: boolean;
  ariaLabel: string;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const optionRefs = useRef<Array<HTMLLIElement | null>>([]);
  const typeaheadRef = useRef({ query: "", timestamp: 0 });
  const selectedIndex = options.findIndex((option) => option.value === value);
  const safeSelectedIndex = selectedIndex >= 0 ? selectedIndex : 0;
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(safeSelectedIndex);
  const listboxId = `${id}-listbox`;
  const unavailable = disabled || options.length === 0;

  useEffect(() => {
    if (!open) return;
    const closeOutside = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        typeaheadRef.current = { query: "", timestamp: 0 };
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", closeOutside);
    return () => document.removeEventListener("mousedown", closeOutside);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    optionRefs.current[activeIndex]?.scrollIntoView?.({ block: "nearest" });
  }, [activeIndex, open]);

  const openMenu = (index = safeSelectedIndex) => {
    setActiveIndex(index);
    setOpen(true);
  };

  const choose = (index: number) => {
    const option = options[index];
    if (!option) return;
    onChange(option.value);
    typeaheadRef.current = { query: "", timestamp: 0 };
    setOpen(false);
    buttonRef.current?.focus();
  };

  const findTypeaheadMatch = (query: string, startIndex: number, includeStart: boolean) => {
    const normalizedQuery = query.toLocaleLowerCase();
    const initialOffset = includeStart ? 0 : 1;
    for (let offset = initialOffset; offset < options.length + initialOffset; offset += 1) {
      const index = (startIndex + offset) % options.length;
      if (options[index]?.label.toLocaleLowerCase().startsWith(normalizedQuery)) return index;
    }
    return -1;
  };

  const handleKeyDown = (event: ReactKeyboardEvent<HTMLButtonElement>) => {
    if (unavailable) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const direction = event.key === "ArrowDown" ? 1 : -1;
      if (open) {
        setActiveIndex((current) => (current + direction + options.length) % options.length);
      } else {
        openMenu((safeSelectedIndex + direction + options.length) % options.length);
      }
      return;
    }
    if (open && event.key === "Home") {
      event.preventDefault();
      setActiveIndex(0);
      return;
    }
    if (open && event.key === "End") {
      event.preventDefault();
      setActiveIndex(options.length - 1);
      return;
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (open) choose(activeIndex);
      else openMenu();
      return;
    }
    if (event.key === "Escape" && open) {
      event.preventDefault();
      typeaheadRef.current = { query: "", timestamp: 0 };
      setOpen(false);
      return;
    }
    if (event.key === "Tab") {
      typeaheadRef.current = { query: "", timestamp: 0 };
      setOpen(false);
      return;
    }
    if (event.key.length === 1 && !event.altKey && !event.ctrlKey && !event.metaKey) {
      event.preventDefault();
      const now = Date.now();
      const previous = typeaheadRef.current;
      const character = event.key.toLocaleLowerCase();
      const withinWindow = now - previous.timestamp < 700;
      const repeatedSingleCharacter = withinWindow && previous.query === character;
      const query = repeatedSingleCharacter
        ? character
        : `${withinWindow ? previous.query : ""}${character}`;
      typeaheadRef.current = { query, timestamp: now };
      const currentIndex = open ? activeIndex : safeSelectedIndex;
      const matchingIndex = findTypeaheadMatch(query, currentIndex, query.length > 1);
      if (matchingIndex >= 0) {
        if (open) setActiveIndex(matchingIndex);
        else openMenu(matchingIndex);
      }
    }
  };

  const selected = selectedIndex >= 0 ? options[selectedIndex] : undefined;
  return (
    <div ref={rootRef} className={`select-menu ${open ? "is-open" : ""}`}>
      <button
        ref={buttonRef}
        id={id}
        type="button"
        role="combobox"
        value={value}
        aria-label={ariaLabel}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listboxId : undefined}
        aria-activedescendant={open && options.length > 0 ? `${listboxId}-option-${activeIndex}` : undefined}
        disabled={unavailable}
        onClick={() => {
          if (open) {
            typeaheadRef.current = { query: "", timestamp: 0 };
            setOpen(false);
          }
          else openMenu();
        }}
        onKeyDown={handleKeyDown}
      >
        <span>{selected?.label ?? "Select an option"}</span>
        <ChevronDown size={16} aria-hidden="true" />
      </button>
      {open && (
        <ul id={listboxId} role="listbox" aria-label={ariaLabel}>
          {options.map((option, index) => (
            <li
              ref={(node) => { optionRefs.current[index] = node; }}
              id={`${listboxId}-option-${index}`}
              key={option.value}
              role="option"
              aria-selected={option.value === value}
              className={`${index === activeIndex ? "is-active" : ""} ${option.value === value ? "is-selected" : ""}`}
              onMouseEnter={() => setActiveIndex(index)}
              onClick={() => choose(index)}
            >
              <span>{option.label}</span>
              {option.value === value && <Check size={15} aria-hidden="true" />}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function InlineAlert({
  tone = "danger",
  title,
  children,
  action,
}: {
  tone?: "danger" | "info" | "success";
  title: string;
  children?: ReactNode;
  action?: () => void;
}) {
  return (
    <div className={`inline-alert alert-${tone}`} role={tone === "danger" ? "alert" : "status"}>
      {tone === "success" ? <Check size={18} /> : <AlertCircle size={18} />}
      <div>
        <strong>{title}</strong>
        {children && <div className="alert-copy">{children}</div>}
      </div>
      {action && (
        <button type="button" className="icon-text-button" onClick={action}>
          <RotateCw size={14} /> Retry
        </button>
      )}
    </div>
  );
}

export function LoadingBlock({ label = "Loading" }: { label?: string }) {
  return (
    <div className="loading-block" role="status">
      <LoaderCircle className="spin" size={22} />
      <span>{label}</span>
    </div>
  );
}

export function StatusDot({ status }: { status: "online" | "offline" | "checking" }) {
  return <span className={`status-dot status-${status}`} aria-hidden="true" />;
}
