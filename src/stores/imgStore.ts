import { writable } from "svelte/store";
export const uploadedImgBase64 = writable<string | null>(null);
export const uploadedImgFileName = writable<string>("");
export const uploadedImgTargetRes = writable<{ width: number; height: number } | null>(null);