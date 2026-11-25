export type Model = {
  id: string; name: string; architecture: string; dataset: string; task: string; status: string;
  checkpoint: { path: string; epoch: number; bytes: number } | null;
  metrics: Record<string, any> | null; test_samples: number; limitations: string;
}
export type Analysis = {
  id: string; created_at: string; media_type: 'image' | 'video'; filename: string; sha256: string;
  metadata: Record<string, any>; result: Record<string, any>; status: string;
}
export type SystemStatus = {
  python_version: string; platform: string; cpu_name:string; logical_cpu_count:number|null; database_status:string; pytorch_version: string; torchvision_version: string; cuda_available: boolean;
  cuda_version: string | null; gpu_name: string | null; gpu_memory_total_bytes: number | null;
  ram_total_bytes: number | null; ram_available_bytes: number | null;
  disk: { project_root_free_bytes: number; total_bytes: number }; video_prediction_status: string;
}
