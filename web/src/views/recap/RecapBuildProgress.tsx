import { Button } from "@/components/ui/button";
import { RecapManifest } from "@/types/recap";
import { useTranslation } from "react-i18next";
import { estimateRemaining, useNow } from "./recapUtils";

type RecapBuildProgressProps = {
  recap: Pick<
    RecapManifest,
    "status" | "progress" | "message" | "started" | "elapsed_s" | "created"
  >;
  title: string;
  onCancel?: () => void;
};

function clock(seconds: number) {
  const whole = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(whole / 60);
  const rest = whole % 60;
  return `${minutes}:${rest.toString().padStart(2, "0")}`;
}

export default function RecapBuildProgress({
  recap,
  title,
  onCancel,
}: RecapBuildProgressProps) {
  const { t } = useTranslation(["views/recap"]);
  const now = useNow(1000);
  const progress = Math.max(0, Math.min(100, recap.progress ?? 0));
  const waiting = recap.status === "queued";
  const elapsed = recap.started ? now - recap.started : (recap.elapsed_s ?? 0);
  const remaining = waiting ? undefined : estimateRemaining(elapsed, progress);

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-secondary bg-background_alt p-4">
      <div className="flex items-baseline justify-between gap-2">
        <div className="text-sm font-medium">{title}</div>
        <div className="text-sm tabular-nums text-muted-foreground">
          {waiting ? t("build.waiting") : `${progress}%`}
        </div>
      </div>
      <div
        className="h-2 overflow-hidden rounded-full bg-secondary"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={progress}
      >
        <div
          className="h-full rounded-full bg-selected transition-[width] duration-700"
          style={{ width: `${waiting ? 0 : Math.max(progress, 2)}%` }}
        />
      </div>
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span className="truncate">
          {waiting
            ? t("build.waitingDetail")
            : recap.message || t("build.starting")}
        </span>
        <span className="tabular-nums">
          {!waiting && t("build.elapsed", { time: clock(elapsed) })}
          {remaining !== undefined &&
            ` · ${t("build.remaining", {
              time: clock(Math.max(remaining, 5)),
            })}`}
        </span>
      </div>
      {onCancel && (
        <div>
          <Button size="sm" variant="outline" onClick={onCancel}>
            {t("cancel")}
          </Button>
        </div>
      )}
    </div>
  );
}
