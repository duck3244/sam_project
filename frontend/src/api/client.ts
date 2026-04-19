import axios from "axios";

export const api = axios.create({
  baseURL: "/api",
  timeout: 300_000,
});

export interface HealthResponse {
  ok: boolean;
  domain: string;
  classes: string[];
  sam_models: string[];
}

export interface SessionCreateResponse {
  session_id: string;
  image_count: number;
}

export interface ImageSummary {
  image_id: string;
  filename: string;
  width: number;
  height: number;
  n_labels: number;
  uncertain: boolean;
  reviewed: boolean;
  priority: number;
}

export interface LabelEntry {
  cls_id: number;
  cls_name: string;
  polygon: number[];
  sam_score: number | null;
  det_conf: number | null;
  bbox: number[] | null;
}

export interface ImageLabels {
  session_id: string;
  image_id: string;
  width: number;
  height: number;
  labels: LabelEntry[];
}

export interface RunRequest {
  sam_model: "vit_b" | "vit_h";
  det_conf: number;
  epsilon?: number;
  max_image_size?: number;
  multi_contour?: boolean;
  min_area_ratio?: number;
}

export interface SessionStats {
  session_id: string;
  image_count: number;
  labeled_count: number;
  total_labels: number;
  class_distribution: Record<string, number>;
  uncertain_queue: number;
}

export const health = () => api.get<HealthResponse>("/health").then((r) => r.data);

export const createSession = () =>
  api.post<SessionCreateResponse>("/session").then((r) => r.data);

export const uploadImages = (sessionId: string, files: File[]) => {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  return api
    .post<{ session_id: string; image_ids: string[] }>(
      `/session/${sessionId}/images`,
      fd,
      { headers: { "Content-Type": "multipart/form-data" } }
    )
    .then((r) => r.data);
};

export const listImages = (sessionId: string, sortByPriority = false) =>
  api
    .get<ImageSummary[]>(`/session/${sessionId}/images`, {
      params: { sort_by_priority: sortByPriority },
    })
    .then((r) => r.data);

export const getLabels = (sessionId: string, imageId: string) =>
  api
    .get<ImageLabels>(`/session/${sessionId}/images/${imageId}/labels`)
    .then((r) => r.data);

export const setLabels = (sessionId: string, imageId: string, body: ImageLabels) =>
  api
    .post<ImageLabels>(`/session/${sessionId}/images/${imageId}/labels`, body)
    .then((r) => r.data);

export const runPipeline = (sessionId: string, req: RunRequest) =>
  api
    .post<{ task_id: string }>(`/pipeline/${sessionId}/run`, req)
    .then((r) => r.data);

export const getStats = (sessionId: string) =>
  api.get<SessionStats>(`/session/${sessionId}/stats`).then((r) => r.data);

export const imageFileUrl = (sessionId: string, imageId: string) =>
  `/api/session/${sessionId}/images/${imageId}/file`;

export const exportZipUrl = (sessionId: string) =>
  `/api/export/${sessionId}?format=yolo-seg`;

export const refineMask = (body: {
  session_id: string;
  image_id: string;
  label_index?: number;
  bbox?: number[];
  points?: number[][];
  point_labels?: number[];
  cls_id?: number;
}) =>
  api
    .post<{ polygon: number[]; sam_score: number }>("/refine", body)
    .then((r) => r.data);
