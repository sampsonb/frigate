export type RecapCategory = {
  key: string;
  name: string;
  color: string;
  count: number;
};

export type RecapTrack = {
  event_id: string;
  clip_event_id: string;
  cat: string;
  label: string;
  text: string;
  start_time: number;
  out_start: number;
  length: number;
  label_box: [number, number, number, number];
  suffix: string | null;
  link_event_id: string | null;
};

export type RecapManifest = {
  id: string;
  camera: string;
  status: string;
  progress?: number;
  message?: string;
  created?: number;
  after?: number;
  before?: number;
  seconds?: number;
  event_count?: number;
  fps?: number;
  width?: number;
  height?: number;
  frame_count?: number;
  source?: "local" | "archive";
  tracks?: RecapTrack[];
  categories?: RecapCategory[];
};

export type RecapSummary = {
  id: string;
  camera: string;
  status: string;
  progress: number;
  message: string;
  created?: number;
  after?: number;
  before?: number;
  seconds?: number;
  event_count?: number;
  width?: number;
  height?: number;
  source?: "local" | "archive";
  reason?: string;
  categories?: RecapCategory[];
};

export type RecapClipSource = {
  source: "frigate" | "archive";
  camera?: string | null;
  clip?: string | null;
  snapshot?: string | null;
  message?: string;
};
