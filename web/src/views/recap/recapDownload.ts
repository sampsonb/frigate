import { baseUrl } from "@/api/baseUrl";
import { useCallback, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";

export type DownloadState = "idle" | "preparing" | "ready";

export type DownloadRequest = {
  // Identifies one file, so a rebuilt rolling recap is fetched again.
  key: string;
  // Serves the file to save (an attachment with a readable name).
  url: string;
  name: string;
  // Plays the video. Opened in its own tab from inside another page's frame.
  openUrl?: string;
};

export function recapVideoUrl(id: string, version?: number) {
  return `${baseUrl}api/recap/${id}/video.mp4${version ? `?v=${version}` : ""}`;
}

/** Same recap, served as a file to save under ``name``. */
export function recapDownloadUrl(id: string, name: string, version?: number) {
  const params = new URLSearchParams({ download: "1", name });
  if (version) {
    params.set("v", String(version));
  }
  return `${baseUrl}api/recap/${id}/video.mp4?${params.toString()}`;
}

/** A file name the Files app, Photos, and Finder all accept. */
export function safeFileName(text: string) {
  const cleaned = text
    .replace(/:/g, ".")
    .replace(/[\\/*?"<>|]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return `${cleaned || "recap"}.mp4`;
}

export function recapFileName(camera: string, span: string) {
  return safeFileName(`${camera} recap ${span}`);
}

/** "Sat Oct 10 12:35 PM to 6:35 PM", with the day even when it is one day. */
export function recapSpanLabel(
  times: {
    day: (epoch?: number) => string;
    span: (after?: number, before?: number) => string;
    stamp: (epoch?: number) => string;
  },
  after?: number,
  before?: number,
  fallback?: number,
) {
  if (!after || !before) {
    return times.stamp(fallback);
  }
  const span = times.span(after, before);
  return times.day(after) === times.day(before)
    ? `${times.day(after)} ${span}`
    : span;
}

function canShareFiles() {
  if (
    typeof navigator === "undefined" ||
    typeof navigator.share !== "function" ||
    typeof navigator.canShare !== "function"
  ) {
    return false;
  }
  try {
    return navigator.canShare({
      files: [new File([new Blob()], "recap.mp4", { type: "video/mp4" })],
    });
  } catch {
    return false;
  }
}

function inFrame() {
  try {
    return window.self !== window.top;
  } catch {
    return true;
  }
}

/**
 * Save a recap video from wherever Frigate is open.
 *
 * Where the share sheet is allowed (an HTTPS page on iPhone or iPad), the
 * file is fetched and handed to it, so Save Video and Save to Files work.
 * Inside another page's frame, such as Home Assistant's webpage card, a
 * download is blocked and an HTTP page has no share sheet. A window opened
 * from that frame inherits its sandbox, so a download there can be blocked
 * too. The video plays in its own tab instead: Safari's Share > Save Video,
 * or a desktop browser's Save As, keeps it. Anywhere else it downloads.
 */
export function useRecapDownload() {
  const { t } = useTranslation(["views/recap"]);
  const [states, setStates] = useState<Record<string, DownloadState>>({});
  // One file at a time. A recap is 5 to 20 MB.
  const cached = useRef<{ key: string; file: File }>();

  const setState = useCallback((key: string, state: DownloadState) => {
    setStates((current) => ({ ...current, [key]: state }));
  }, []);

  const share = useCallback(
    async (key: string, file: File) => {
      try {
        await navigator.share({ files: [file], title: file.name });
        setState(key, "idle");
      } catch (error) {
        const reason = error instanceof DOMException ? error.name : "";
        if (reason === "AbortError") {
          setState(key, "idle");
          return;
        }
        if (reason === "NotAllowedError") {
          // The tap expired while the video downloaded. The next tap
          // shares the file that is already here.
          setState(key, "ready");
          toast.message(t("downloadReady"), { position: "top-center" });
          return;
        }
        setState(key, "idle");
        toast.error(t("downloadFailed"), { position: "top-center" });
      }
    },
    [setState, t],
  );

  const download = useCallback(
    async ({ key, url, name, openUrl }: DownloadRequest) => {
      if (canShareFiles()) {
        if (cached.current?.key === key) {
          await share(key, cached.current.file);
          return;
        }
        setState(key, "preparing");
        try {
          const response = await fetch(url);
          if (!response.ok) {
            throw new Error(String(response.status));
          }
          const file = new File([await response.blob()], name, {
            type: "video/mp4",
          });
          cached.current = { key, file };
          await share(key, file);
        } catch {
          setState(key, "idle");
          toast.error(t("downloadFailed"), { position: "top-center" });
        }
        return;
      }
      if (inFrame()) {
        window.open(openUrl ?? url, "_blank", "noopener");
        return;
      }
      const link = document.createElement("a");
      link.href = url;
      link.download = name;
      link.rel = "noopener";
      document.body.appendChild(link);
      link.click();
      link.remove();
    },
    [setState, share, t],
  );

  const stateOf = useCallback(
    (key: string): DownloadState => states[key] ?? "idle",
    [states],
  );

  return { download, stateOf };
}
